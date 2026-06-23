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
    set_pool_description, get_pool_ids, update_pool_translate, set_dictionary_studying,
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
from models import DictCreate, AddWords, ImportDict, PoolAdd, PoolToDict, WordOverride, ResultBody, MoveWords, RefineWords, AiWordsBody, StudyingBody

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


@router.post("/words/{dw_id}/result")
async def word_result(dw_id: int, body: ResultBody, user=Depends(get_current_user)):
    return await record_result(user["id"], dw_id, body.correct, mode=body.mode, elapsed=body.elapsed, direction=body.direction)


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
