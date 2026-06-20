import json
import random
from fastapi import APIRouter, Depends, HTTPException
from config import logger
import errors
import notify
from db import (
    get_user_data, create_dictionary, rename_dictionary, delete_dictionary, add_word_to_dict,
    delete_dict_word, move_dict_word, set_word_override, record_result, get_dict_word,
    get_or_create_pool, get_pool_id, get_pool_candidates, delete_pool_word,
    set_pool_description, get_pool_ids, update_pool_translate,
)
from auth import get_current_user
from activity import mark_activity
from llm import (
    generate_words, ensure_embeddings, persist_pool, ask_json, DESC_SCHEMA,
    ranked_pool, text_enabled, TOPIC_KEYS, CEFR_LEVELS, refine_translations,
)
from tts import schedule_tts
from task import description_task, desc_user_prompt
from autofill import ai_game_words
from models import DictCreate, AddWords, ImportDict, PoolAdd, PoolToDict, WordOverride, ResultBody, MoveWords, RefineWords, AiWordsBody

router = APIRouter()


@router.post("/games/ai_words")
async def games_ai_words(body: AiWordsBody, user=Depends(get_current_user)):
    """AI-подбор слов для одиночной игры по уровню/теме (как в онлайне). Возвращает набор
    [{no, translate}] — слова попадают в общий пул и обогащаются фоном."""
    mark_activity()
    count = max(3, min(30, body.count or 10))
    words = await ai_game_words(body.lang or "ru", body.level or None, body.topic or None, count)
    return {"words": [{"no": w["norwegian"], "translate": w["translate"]} for w in words]}


def c2pos(c):
    return c["data"].get("part_of_speech", "")


# --- User data (server-backed) ---
@router.get("/data")
async def data(user=Depends(get_current_user)):
    return await get_user_data(user["id"])


@router.post("/dictionaries")
async def create_dict(body: DictCreate, user=Depends(get_current_user)):
    res = await create_dictionary(user["id"], body.name)
    if res.get("error"):
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@router.delete("/dictionaries/{dict_id}")
async def remove_dict(dict_id: int, user=Depends(get_current_user)):
    res = await delete_dictionary(user["id"], dict_id)
    if res.get("error"):
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@router.post("/dictionaries/{dict_id}/rename")
async def rename_dict(dict_id: int, body: DictCreate, user=Depends(get_current_user)):
    res = await rename_dictionary(user["id"], dict_id, body.name)
    if res.get("error"):
        raise HTTPException(status_code=400, detail=res["error"])
    return res


@router.post("/dictionaries/{dict_id}/words")
async def add_words(dict_id: int, body: AddWords, user=Depends(get_current_user)):
    mark_activity()
    normalized, cached = await generate_words(body.prompt, body.model)
    errs, words, pairs = [], [], []
    added_words, dup_words = [], []  # добавлено в словарь / уже там было
    for item in normalized:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            errs.append(item["error"]); continue
        if not item.get("word"):
            continue
        pool_id = await get_or_create_pool(item["word"], item)
        if pool_id:
            pairs.append((pool_id, item))
            res = await add_word_to_dict(user["id"], dict_id, pool_id)
            (added_words if (res.get("id") and not res.get("duplicate")) else dup_words).append(item["word"])
            words.append(item["word"])
    await ensure_embeddings(pairs)  # эмбеддинги добавленных слов одним запросом
    schedule_tts(words)  # озвучку добавленных слов ставим в очередь сразу
    # Разбивка в ленту: почему добавилось не всё (часть слов уже была в словаре).
    src = "из кэша" if cached else "сгенерировано"
    msg = (f"📥 «{(body.prompt or '')[:40]}» ({src}): получено {len(added_words) + len(dup_words)} · "
           f"добавлено {len(added_words)} · уже в словаре {len(dup_words)}")
    if added_words:
        msg += f"\n  ➕ {', '.join(added_words)}"
    if dup_words:
        msg += f"\n  ↩️ уже были: {', '.join(dup_words)}"
    if errs:
        msg += f"\n  ⚠️ ошибок: {len(errs)}"
    notify.feed(msg)
    return {"added": len(added_words), "errors": errs, "cached": cached,
            "duplicates": len(dup_words), "added_words": added_words, "duplicate_words": dup_words}


@router.post("/dictionaries/{dict_id}/add_pool")
async def add_pool(dict_id: int, body: PoolAdd, user=Depends(get_current_user)):
    """Добавить в словарь уже существующее в общем пуле слово (автокомплит) — без ИИ."""
    pid = await get_pool_id(body.norwegian)
    if not pid:
        raise HTTPException(status_code=404, detail="Word not in pool")
    res = await add_word_to_dict(user["id"], dict_id, pid)
    schedule_tts([body.norwegian])  # на случай если у слова ещё нет озвучки
    return {"added": 1 if (res.get("id") and not res.get("duplicate")) else 0, "duplicate": res.get("duplicate", False)}


@router.post("/dictionaries/from_pool")
async def dict_from_pool(body: PoolToDict, user=Depends(get_current_user)):
    """Создать новый словарь и добавить в него ВСЕ слова пула, подходящие под фильтр."""
    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Empty name")
    res = await create_dictionary(user["id"], name)
    dict_id = res.get("id")
    if not dict_id:
        raise HTTPException(status_code=400, detail=res.get("error", "Dictionary already exists"))
    topic_list = [t for t in (body.topics or []) if t in TOPIC_KEYS]
    lvl = body.level if body.level in CEFR_LEVELS else None
    ids = await get_pool_ids(body.q, topic_list, lvl)
    added = 0
    for pid in ids:
        r = await add_word_to_dict(user["id"], dict_id, pid)
        if r.get("id") and not r.get("duplicate"):
            added += 1
    return {"dict_id": dict_id, "name": name, "added": added, "total": len(ids)}


@router.post("/dictionaries/import")
async def import_dict(body: ImportDict, user=Depends(get_current_user)):
    res = await create_dictionary(user["id"], body.name)
    dict_id = res.get("id")
    if not dict_id:
        # словарь существует — найдём его через данные
        data = await get_user_data(user["id"])
        match = next((d for d in data["dictList"] if d["dictName"] == body.name.strip()), None)
        if not match:
            raise HTTPException(status_code=400, detail=res.get("error", "Import failed"))
        dict_id = match["id"]
    added = 0
    for w in body.words:
        tr = w.get("translate", {})
        no = (tr.get("no") or [w.get("word")])[0] if (tr.get("no") or w.get("word")) else None
        if not no:
            continue
        item = {"word": no, "translate": {**tr, "no": [no]}, "part_of_speech": w.get("part_of_speech", "")}
        pool_id = await get_or_create_pool(no, item)
        if pool_id:
            r = await add_word_to_dict(user["id"], dict_id, pool_id)
            if r.get("id") and not r.get("duplicate"):
                added += 1
    return {"added": added, "dict_id": dict_id}


@router.delete("/words/{dw_id}")
async def remove_word(dw_id: int, user=Depends(get_current_user)):
    return await delete_dict_word(user["id"], dw_id)


@router.post("/words/move")
async def move_words(body: MoveWords, user=Depends(get_current_user)):
    """Перенести выбранные слова (по dw_id) в другой словарь пользователя."""
    moved = 0
    for dw_id in body.ids:
        res = await move_dict_word(user["id"], dw_id, body.dict_id)
        if res.get("moved"):
            moved += 1
    return {"moved": moved, "total": len(body.ids)}


@router.post("/words/refine")
async def refine_words(body: RefineWords, user=Depends(get_current_user)):
    """Уточнить перевод группы выбранных слов (одинаковый/неточный перевод) через LLM.
    Правит перевод на язык body.lang в общем пуле (для всех)."""
    if not text_enabled():
        raise HTTPException(status_code=503, detail="LLM not configured")
    lang = body.lang
    items, pool_map = [], {}
    for dw_id in body.ids:
        dw = await get_dict_word(user["id"], dw_id)
        if not dw:
            continue
        data = json.loads(dw["data"]) if dw["data"] else {}
        cur = (data.get("translate", {}) or {}).get(lang) or []
        no = dw["norwegian"]
        items.append({"word": no, "current": cur})
        pool_map[no.lower()] = (dw["pool_id"], data)
    if len(items) < 2:
        raise HTTPException(status_code=400, detail="Select at least two words")
    mark_activity()
    try:
        refined = await refine_translations(items, lang)
    except Exception as e:
        info = errors.report(e, "refine_translations")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    updated = 0
    for no_l, (pool_id, data) in pool_map.items():
        new_tr = refined.get(no_l)
        if not new_tr:
            continue
        translate = data.get("translate", {}) or {}
        if translate.get(lang) == new_tr:
            continue
        translate[lang] = new_tr
        await update_pool_translate(pool_id, translate)
        updated += 1
    return {"updated": updated, "total": len(items)}


@router.patch("/words/{dw_id}")
async def edit_word(dw_id: int, body: WordOverride, user=Depends(get_current_user)):
    override = {}
    if body.translate is not None:
        override["translate"] = body.translate
    if body.part_of_speech is not None:
        override["part_of_speech"] = body.part_of_speech
    return await set_word_override(user["id"], dw_id, override)


@router.post("/words/{dw_id}/result")
async def word_result(dw_id: int, body: ResultBody, user=Depends(get_current_user)):
    return await record_result(user["id"], dw_id, body.correct, mode=body.mode, elapsed=body.elapsed)


@router.get("/words/{dw_id}/description")
async def word_description(dw_id: int, model: str = None, user=Depends(get_current_user)):
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    if dw["description"]:
        return {"description": json.loads(dw["description"])}
    data = json.loads(dw["data"]) if dw["data"] else {}
    if dw["override"]:
        data = {**data, **json.loads(dw["override"])}
    try:
        desc = await ask_json(description_task, desc_user_prompt(dw["norwegian"], data), DESC_SCHEMA,
                              purpose="user", label="описание слова", model=model)
    except Exception as e:
        info = errors.report(e, "word_description")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    if not isinstance(desc, dict):
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    description = desc.get("description", desc)
    await set_pool_description(dw["pool_id"], description)
    return {"description": description}


@router.get("/words/{dw_id}/distractors")
async def distractors(dw_id: int, n: int = 3, mode: str = "no2int", lang: str = "ru", user=Depends(get_current_user)):
    """Неправильные варианты для режима «выбор»: семантически близкие (по эмбеддингам),
    иначе — той же части речи. mode: no2int (ответ — перевод на lang) | int2no (ответ — норвежское)."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    target = json.loads(dw["data"]) if dw["data"] else {}
    if dw["override"]:
        ov = json.loads(dw["override"]); target = {**target, **ov}
    target_pos = target.get("part_of_speech", "")

    def answer_of(data, norwegian):
        if mode == "int2no":
            return norwegian
        tr = (data.get("translate", {}) or {}).get(lang) or []
        return tr[0] if tr else None

    correct = (dw["norwegian"] if mode == "int2no" else answer_of(target, dw["norwegian"]))
    correct_l = (correct or "").strip().lower()

    ordered = await ranked_pool(dw["embedding"], dw["norwegian"], 40)  # семантически близкие (ANN/brute)
    if not ordered:
        # нет эмбеддинга/индекса — случайные той же части речи
        cands = [c for c in await get_pool_candidates() if c["norwegian"] != dw["norwegian"]]
        same = [c for c in cands if c["data"].get("part_of_speech") == target_pos]
        other = [c for c in cands if c["data"].get("part_of_speech") != target_pos]
        random.shuffle(same); random.shuffle(other)
        ordered = same + other

    out, seen = [], {correct_l}
    for c in ordered:
        a = answer_of(c["data"], c["norwegian"])
        if a and a.strip().lower() not in seen:
            out.append(a); seen.add(a.strip().lower())
        if len(out) >= n:
            break
    return {"distractors": out}


@router.post("/words/{dw_id}/report")
async def report_word(dw_id: int, user=Depends(get_current_user)):
    """Пометить слово как неправильное: удалить из общего пула (у всех) и перегенерировать заново."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    norwegian = dw["norwegian"]
    dict_id = dw["dict_id"]
    await delete_pool_word(norwegian)  # убираем из пула у всех + чистим кэш

    regenerated, new_word = False, None
    if text_enabled():
        try:
            mark_activity()
            normalized, _ = await generate_words(norwegian, None)
            await persist_pool(normalized)
            for it in normalized:
                if isinstance(it, dict) and not it.get("error") and it.get("word"):
                    pid = await get_pool_id(it["word"])
                    if pid:
                        await add_word_to_dict(user["id"], dict_id, pid)
                        regenerated, new_word = True, it["word"]
                        schedule_tts([it["word"]])
                        break
        except Exception as e:
            logger.warning(f"report regen failed: {e}")
    return {"removed": True, "regenerated": regenerated, "word": new_word}


@router.get("/words/{dw_id}/synonyms")
async def synonyms(dw_id: int, n: int = 5, lang: str = "ru", user=Depends(get_current_user)):
    """Близкие по смыслу слова из общего пула (по эмбеддингам). Без ключа эмбеддингов — пусто."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    if not dw["embedding"]:
        return {"synonyms": []}
    ordered = await ranked_pool(dw["embedding"], dw["norwegian"], n)
    out = []
    for c in ordered[:n]:
        tr = (c["data"].get("translate", {}) or {}).get(lang) or []
        out.append({"word": c["norwegian"], "translate": tr})
    return {"synonyms": out}
