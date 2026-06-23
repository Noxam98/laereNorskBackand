from fastapi import APIRouter, Depends, HTTPException, Response
from config import logger
import errors
import asyncio
import json
from datetime import datetime
from db import (
    normalize_word, get_pool_tts, set_pool_tts, get_pool_id, get_pool_by_id,
    set_pool_description, get_pool_list, delete_pool_word, pool_missing_description,
    search_pool, get_pool_topics_counts, get_pool_level_counts, get_pool_facets, get_pool_meta, get_pool_stats, get_usage_like,
    get_cached_query, cache_query, set_cached_query, update_pool_word, replace_pool_word,
    pending_words, pending_count, set_word_approval,
)
from auth import get_current_user, get_admin_user
from activity import mark_activity
from tts import synth_tts, _tts_lock
from llm import TOPIC_KEYS, CEFR_LEVELS, ask_json, DESC_SCHEMA, DIFF_SCHEMA, REVIEW_SCHEMA, LANG_NAMES, ranked_pool, normalize_word_item, generate_words, persist_pool, text_enabled, embed_text
from task import description_task, desc_user_prompt
from models import RedescribeBody, RediffBody, PoolEditBody, AskBody
import runtime
import storage

router = APIRouter()

_TTS_HEADERS = {"Cache-Control": "public, max-age=604800"}
_TRANSLATION_LANGS = {"ru", "uk", "en", "pl", "lt"}  # языки озвучки переводов


async def _tts_translation(text: str, lang: str):
    """Озвучка перевода нужным голосом, кэш в объектном хранилище (Tigris)."""
    okey = storage.key_for(lang, text)
    data = await storage.get_object(okey)
    if data:
        return Response(content=data, media_type="audio/mpeg", headers=_TTS_HEADERS)
    async with _tts_lock:
        data = await storage.get_object(okey)  # могли сгенерить, пока ждали очередь
        if data:
            return Response(content=data, media_type="audio/mpeg", headers=_TTS_HEADERS)
        try:
            mp3 = await synth_tts(text, lang)
        except Exception as e:
            logger.warning(f"tts({lang}) failed: {e}")
            raise HTTPException(status_code=502, detail="TTS provider error")
        if not mp3:
            raise HTTPException(status_code=502, detail="No audio")
        await storage.put_object(okey, mp3)
        return Response(content=mp3, media_type="audio/mpeg", headers=_TTS_HEADERS)


@router.get("/tts")
async def tts(word: str, lang: str = None):
    """Аудио произношения. Без lang (или nb) — норвежское слово (кэш в пуле БД).
    lang=ru/uk/en/pl/lt — озвучка перевода нужным голосом (кэш в Tigris). Публичный."""
    text = (word or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="word is required")

    if lang and lang in _TRANSLATION_LANGS:
        return await _tts_translation(text, lang)

    # Норвежский — как было.
    key = normalize_word(text)
    cached = await get_pool_tts(key)
    if cached:
        return Response(content=bytes(cached), media_type="audio/mpeg", headers=_TTS_HEADERS)

    async with _tts_lock:
        cached = await get_pool_tts(key)  # могли сгенерить, пока ждали очередь
        if cached:
            return Response(content=bytes(cached), media_type="audio/mpeg", headers=_TTS_HEADERS)
        try:
            mp3 = await synth_tts(key)
        except Exception as e:
            logger.warning(f"tts failed: {e}")
            raise HTTPException(status_code=502, detail="TTS provider error")
        if not mp3:
            raise HTTPException(status_code=502, detail="No audio")
        if await get_pool_id(key):
            await set_pool_tts(key, mp3)
        return Response(content=mp3, media_type="audio/mpeg", headers=_TTS_HEADERS)


@router.post("/pool/{word}/revoice")
async def pool_revoice(word: str, user=Depends(get_current_user)):
    """Переозвучить норвежское слово — перегенерировать аудио и перезаписать кэш."""
    key = normalize_word(word)
    if not await get_pool_id(key):
        raise HTTPException(status_code=404, detail="Not in pool")
    mark_activity()
    async with _tts_lock:
        try:
            mp3 = await synth_tts(key)
        except Exception as e:
            logger.warning(f"revoice failed: {e}")
            raise HTTPException(status_code=502, detail="TTS provider error")
        if not mp3:
            raise HTTPException(status_code=502, detail="No audio")
        await set_pool_tts(key, mp3)
    return {"ok": True}


# --- Shared pool ---
@router.get("/pool")
async def pool(q: str = None, limit: int = 60, offset: int = 0,
              topics: str = None, level: str = None,
              sort: str = "alpha", order: str = "asc", missing: str = None, pos: str = None,
              lang: str = None, user=Depends(get_current_user)):
    topic_list = [t for t in (topics.split(",") if topics else []) if t in TOPIC_KEYS]
    lvl = level if level in CEFR_LEVELS else None
    srt = sort if sort in ("alpha", "level", "added", "freq", "relevance") else "alpha"
    res = await get_pool_list(limit, offset, q, topic_list, lvl, srt, order, missing, pos, user_id=user["id"], lang=lang, embed_fn=embed_text)
    res["facets"] = await get_pool_facets(q, topic_list, lvl, lang=lang, user_id=user["id"])  # счётчики под текущий фильтр
    return res


@router.get("/pool/topics")
async def pool_topics(user=Depends(get_current_user)):
    return {"topics": await get_pool_topics_counts(), "levels": await get_pool_level_counts()}


@router.delete("/admin/pool/{word}")
async def admin_delete_word(word: str, user=Depends(get_admin_user)):
    """Полностью удалить слово из общего пула (у всех + кэш + ANN-индекс). Только админ."""
    await delete_pool_word(word)
    return {"ok": True}


@router.get("/admin/pending")
async def admin_pending(limit: int = 300, offset: int = 0, user=Depends(get_admin_user)):
    """Слова на модерации (личные расширения юзеров) — для админа."""
    return {"words": await pending_words(limit, offset), "count": await pending_count()}


@router.post("/admin/pending/{pool_id}/approve")
async def admin_pending_approve(pool_id: int, user=Depends(get_admin_user)):
    """Одобрить слово → в общую базу (видно всем)."""
    return await set_word_approval(pool_id, 1)


@router.post("/admin/pending/{pool_id}/reject")
async def admin_pending_reject(pool_id: int, user=Depends(get_admin_user)):
    """Отклонить → остаётся приватным у автора (из общей базы скрыто, у автора работает)."""
    return await set_word_approval(pool_id, 2)


@router.get("/admin/embeddings")
async def admin_embeddings(limit: int = 1000, offset: int = 0, user=Depends(get_admin_user)):
    """Постраничная выгрузка векторов пула: {"items": [[id, hex(embedding)], ...]}. Только админ."""
    from db import get_pool_embeddings_page
    return {"items": await get_pool_embeddings_page(limit, offset)}


@router.get("/admin/dedup_status")
async def admin_dedup_status(user=Depends(get_admin_user)):
    """Прогресс фонового дедупа: сколько слов проверено из всех с эмбеддингом + на паузе ли."""
    from db import dedup_progress
    import runtime
    done, total = await dedup_progress()
    return {"checked": done, "total": total, "remaining": total - done,
            "paused": runtime.PAUSED.get("dedup", False)}


@router.get("/admin/pool_meta")
async def admin_pool_meta(user=Depends(get_admin_user)):
    """Лёгкие метаданные пула для дедупа: [[id, norwegian, {lng:tr}, pop], ...]. Только админ."""
    from db import get_pool_meta_all
    return {"items": await get_pool_meta_all()}


@router.post("/admin/describe_all")
async def admin_describe_all(user=Depends(get_admin_user)):
    """Запустить фоновую пакетную догенерацию описаний для всех слов без описания."""
    from autofill import describe_all_task
    pending = len(await pool_missing_description(1000000))
    asyncio.create_task(describe_all_task())
    return {"pending": pending, "started": True}


@router.get("/admin/control")
async def admin_get_control(user=Depends(get_admin_user)):
    """Текущее состояние пауз фоновых задач (autofill/embed/describe)."""
    return {"paused": runtime.PAUSED}


@router.post("/admin/control/{key}")
async def admin_set_control(key: str, paused: bool, user=Depends(get_admin_user)):
    """Поставить/снять паузу фоновой задачи. key: autofill|embed|describe."""
    if key not in runtime.PAUSED:
        raise HTTPException(status_code=400, detail="unknown control key")
    runtime.PAUSED[key] = bool(paused)
    await runtime.persist()  # сохранить в БД, чтобы пережило рестарт/передеплой
    logger.info(f"admin: {key} paused={runtime.PAUSED[key]}")
    return {"paused": runtime.PAUSED}


@router.get("/admin/stats")
async def admin_stats(user=Depends(get_admin_user)):
    """Техническая статистика проекта (только для админа)."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    return {
        "pool": await get_pool_stats(),
        "topics": await get_pool_topics_counts(),
        "levels": await get_pool_level_counts(),
        "usageToday": await get_usage_like(today),
    }


@router.get("/pool/search")
async def pool_search(q: str, limit: int = 10, lang: str = None, user=Depends(get_current_user)):
    return {"results": await search_pool(q, limit, lang)}


@router.post("/pool/generate")
async def pool_generate(body: dict, user=Depends(get_current_user)):
    """Сгенерировать новое слово (LLM) и положить в пул — для слов, которых нет ни в пуле, ни
    в лексиконе. Если слово уже есть — просто вернуть его. Возвращает {word, pool_id, translate, generated}."""
    word = (body.get("word") or "").strip()
    if not word:
        raise HTTPException(status_code=400, detail="no word")
    pid = await get_pool_id(normalize_word(word))
    if pid:
        p = await get_pool_by_id(pid)
        return {"word": word, "pool_id": pid, "generated": False,
                "translate": (p or {}).get("translate", {}) if isinstance(p, dict) else {}}
    if not text_enabled():
        raise HTTPException(status_code=503, detail="generation disabled")
    mark_activity()
    try:
        normalized, _ = await generate_words(word, None)
        # модерация: новые слова от юзера → в его личное расширение (approved=0), не в общую базу
        await persist_pool(normalized, created_by=user["id"], approved=0)
    except Exception as e:
        logger.warning(f"pool generate '{word}' failed: {e}")
        raise HTTPException(status_code=502, detail="generation failed")
    # уведомить админов (фоном, не блокируя ответ): пуш при закрытом приложении, тост — при открытом
    async def _notify_mods():
        try:
            from webpush import notify_moderators
            await notify_moderators(await pending_count())
        except Exception:
            pass
    asyncio.create_task(_notify_mods())
    for it in normalized:
        if isinstance(it, dict) and not it.get("error") and it.get("word"):
            npid = await get_pool_id(it["word"], it.get("part_of_speech", ""))
            if npid:
                return {"word": it["word"], "pool_id": npid, "generated": True,
                        "translate": it.get("translate", {})}
    raise HTTPException(status_code=502, detail="generation produced nothing")


@router.get("/pool/{word}/description")
async def pool_description(word: str, model: str = None, user=Depends(get_current_user)):
    """Описание слова из общего пула (как в личном словаре): есть — отдаём, нет — генерим и кэшируем."""
    pid = await get_pool_id(word)
    if not pid:
        raise HTTPException(status_code=404, detail="Not in pool")
    p = await get_pool_by_id(pid)
    if p and p.get("description"):
        return {"description": p["description"]}
    mark_activity()
    desc = await ask_json(description_task, desc_user_prompt(normalize_word(word), p.get("data") if p else None), DESC_SCHEMA,
                          purpose="user", label="описание слова", model=model)
    if not isinstance(desc, dict):
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    description = desc.get("description", desc)
    await set_pool_description(pid, description)
    return {"description": description}


@router.post("/pool/{word}/redescribe")
async def pool_redescribe(word: str, body: RedescribeBody, user=Depends(get_current_user)):
    """Перегенерировать описание слова (при неверном) с учётом подсказки пользователя
    о правильном значении. Перезаписывает кэш описания в общем пуле."""
    pid = await get_pool_id(word)
    if not pid:
        raise HTTPException(status_code=404, detail="Not in pool")
    hint = (body.hint or "").strip()
    mark_activity()
    p = await get_pool_by_id(pid)
    extra = ("ВАЖНО: предыдущее описание было неверным. Правильное значение/уточнение "
             f"от пользователя (учти обязательно): {hint}") if hint else ""
    user_prompt = desc_user_prompt(normalize_word(word), p.get("data") if p else None, extra)
    desc = await ask_json(description_task, user_prompt, DESC_SCHEMA, purpose="user", label="переописание слова")
    if not isinstance(desc, dict):
        raise HTTPException(status_code=502, detail="No JSON")
    description = desc.get("description", desc)
    await set_pool_description(pid, description)
    return {"description": description}


@router.post("/pool/{word}/ask")
async def pool_ask(word: str, body: AskBody, user=Depends(get_current_user)):
    """Свободный вопрос пользователя о слове — нейросеть отвечает с учётом части речи и
    переводов (контекст значения), кратко, на языке интерфейса."""
    pid = await get_pool_id(word)
    if not pid:
        raise HTTPException(status_code=404, detail="Not in pool")
    q = (body.question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    p = await get_pool_by_id(pid)
    lang_name = LANG_NAMES.get(body.lang, "русский")
    mark_activity()
    sys = ("Ты — дружелюбный преподаватель норвежского (bokmål). Кратко и по делу ответь на "
           f"вопрос пользователя об этом норвежском слове. Отвечай на языке «{lang_name}», 1–5 "
           "предложений, можно с примером. Если вопрос не относится к слову/языку — мягко вернись к теме.")
    ctx = desc_user_prompt(normalize_word(word), p.get("data") if p else None, f"Вопрос пользователя: {q}")
    schema = {"name": "word_answer", "schema": {
        "type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}}
    try:
        res = await ask_json(sys, ctx, schema, purpose="user", label="вопрос о слове")
    except Exception as e:
        info = errors.report(e, "pool_ask")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    return {"answer": (res or {}).get("answer", "") if isinstance(res, dict) else ""}


@router.post("/pool/{word}/edit")
async def pool_edit(word: str, body: PoolEditBody, user=Depends(get_current_user)):
    """Правка слова в общем пуле (норвежское слово + переводы) — меняется для всех.
    Сначала ревью нейросетью: применяем только при approved, иначе возвращаем причину."""
    pid = await get_pool_id(word)
    if not pid:
        raise HTTPException(status_code=404, detail="Not in pool")
    tr = body.translate or {}
    hint = (body.hint or "").strip()
    lang_name = LANG_NAMES.get(body.lang, "русский")
    mark_activity()
    review_sys = (
        "Ты — модератор словаря норвежского (bokmål). Пользователь предлагает правку слова и/или переводов "
        "(может дать подсказку о части речи). Проверь: норвежское слово реально существует (возможна "
        "орфографическая опечатка пользователя). "
        "• Если слово реально (пусть с опечаткой) — approved=true и верни поле word в СТАНДАРТИЗОВАННОМ виде, "
        "как при обычной генерации: исправленное правильное написание (bokmål), part_of_speech "
        "(noun/verb/adjective/adverb/preposition/conjunction/pronoun/determiner/numeral/interjection/phrase; "
        "учти подсказку пользователя), переводы на ru/ukr/en/pl/lt (1-3 варианта). "
        "• Если слова не существует, или это смысловая ошибка, или непонятно — approved=false (word не нужен). "
        f"reason — 1-2 предложения на языке «{lang_name}»: что исправлено / почему отклонено."
    )
    user_prompt = (f"Текущее норвежское слово: >>{normalize_word(word)}<<\n"
                   f"Предлагаемая правка (переводы по языкам; 'no' — норвежское слово):\n{json.dumps(tr, ensure_ascii=False)}"
                   + (f"\nПодсказка пользователя: {hint}" if hint else ""))
    try:
        verdict = await ask_json(review_sys, user_prompt, REVIEW_SCHEMA, purpose="user", label="ревью правки слова")
    except Exception as e:
        errors.report(e, "pool_edit_review")
        raise HTTPException(status_code=502, detail="Review failed")
    approved = bool(verdict.get("approved")) if isinstance(verdict, dict) else False
    reason = (verdict.get("reason") or "") if isinstance(verdict, dict) else ""
    if not approved:
        return {"approved": False, "reason": reason}

    # Стандартизованное слово от нейросети (исправленная орфография). Фолбэк — пользовательский ввод.
    wd = verdict.get("word") if isinstance(verdict.get("word"), dict) else {}
    new_no = (wd.get("word") or "").strip() or (tr.get("no") or [normalize_word(word)])[0]
    item = normalize_word_item({
        "word": new_no,
        "part_of_speech": wd.get("part_of_speech") or "",
        "translate": {k: v for k, v in (wd.get("translate") or {}).items() if v},
    })
    res = await replace_pool_word(word, new_no, item)
    if res.get("error") == "not_found":
        raise HTTPException(status_code=404, detail="Not in pool")
    if res.get("error") == "exists":
        raise HTTPException(status_code=409, detail="Word already exists")
    return {"approved": True, "ok": True, "reason": reason, "norwegian": res.get("norwegian"),
            "no": new_no, "translate": item.get("translate", {}), "part_of_speech": item.get("part_of_speech", "")}


@router.get("/pool/{word}/synonyms")
async def pool_synonyms(word: str, n: int = 5, lang: str = "ru", user=Depends(get_current_user)):
    """Близкие по смыслу слова из пула (по эмбеддингам). Без эмбеддинга — пусто."""
    pid = await get_pool_id(word)
    if not pid:
        raise HTTPException(status_code=404, detail="Not in pool")
    p = await get_pool_by_id(pid)
    if not p or not p.get("embedding"):
        return {"synonyms": []}
    ordered = await ranked_pool(p["embedding"], normalize_word(word), n)
    out = []
    for c in ordered[:n]:
        tr = (c["data"].get("translate", {}) or {}).get(lang) or []
        out.append({"word": c["norwegian"], "translate": tr})
    return {"synonyms": out}


@router.get("/pool/{pool_id}/distractors")
async def pool_distractors(pool_id: int, n: int = 3, mode: str = "no2int", lang: str = "ru", user=Depends(get_current_user)):
    """Неправильные варианты для «выбора» по слову ПУЛА (для системной сессии «Учёбы»):
    семантически близкие по эмбеддингам, иначе той же части речи. mode no2int|int2no."""
    import random
    from db import get_pool_candidates
    p = await get_pool_by_id(pool_id)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    data = p["data"] or {}   # get_pool_by_id уже отдаёт data распарсенным
    target_pos = data.get("part_of_speech", "")
    norwegian = data.get("word") or ((data.get("translate", {}) or {}).get("no") or [""])[0]

    def answer_of(d, no):
        # вернуть до ДВУХ вариантов ответа: (основной, второй|None)
        if mode == "int2no":
            return (no, None)
        tr = (d.get("translate", {}) or {}).get(lang) or []
        return (tr[0] if tr else None, tr[1] if len(tr) > 1 else None)

    # СМЫСЛ цели = все её переводы на язык юзера; дистрактор-синоним (пересечение переводов) исключаем.
    target_mean = {x.strip().lower() for x in ((data.get("translate", {}) or {}).get(lang) or []) if x}
    own = ({(norwegian or "").strip().lower()} | {x.strip().lower() for x in ((data.get("translate", {}) or {}).get("no") or []) if x}) \
        if mode == "int2no" else set(target_mean)
    own.discard("")
    ordered = await ranked_pool(p["embedding"], norwegian, 40) if p.get("embedding") else []
    if not ordered:
        cands = [c for c in await get_pool_candidates() if c["norwegian"] != norwegian]
        same = [c for c in cands if c["data"].get("part_of_speech") == target_pos]
        other = [c for c in cands if c["data"].get("part_of_speech") != target_pos]
        random.shuffle(same); random.shuffle(other)
        ordered = same + other
    out, seen = [], set(own)
    for c in ordered:
        cmean = {x.strip().lower() for x in ((c["data"].get("translate", {}) or {}).get(lang) or []) if x}
        if target_mean and (cmean & target_mean):   # синоним по смыслу — пропускаем
            continue
        a, alt = answer_of(c["data"], c["norwegian"])
        if a and a.strip().lower() not in seen:
            out.append({"w": a, "alt": alt}); seen.add(a.strip().lower())
        if len(out) >= n:
            break
    # обратная совместимость + богатый формат
    return {"distractors": [o["w"] for o in out], "options": out}


_DIFF_LANG_NAMES = {"ru": "русском", "ukr": "украинском", "en": "English", "pl": "polskim", "lt": "lietuvių"}
_DIFF_LANGS = {"ru", "ukr", "en", "pl", "lt"}


def _diff_sys(lang, hint=None):
    sys = (
        "Ты — преподаватель норвежского (bokmål). Объясни РАЗНИЦУ между двумя норвежскими "
        f"словами кратко и по делу на языке: {_DIFF_LANG_NAMES.get(lang, lang)}. Поля: "
        "summary — суть различия одной фразой; when_a — когда употреблять ПЕРВОЕ слово; "
        "when_b — когда употреблять ВТОРОЕ слово; example — один короткий пример "
        "(норвежская фраза + перевод). Весь текст на указанном языке; норвежские слова в "
        "примере оставляй как есть."
    )
    if hint:
        sys += f"\nВАЖНО: предыдущее объяснение было неверным. Уточнение от пользователя (учти обязательно): {hint}"
    return sys


async def _gen_diff(a, b, lang, hint=None):
    try:
        data = await ask_json(_diff_sys(lang, hint), f"Первое слово: >>{a}<<\nВторое слово: >>{b}<<", DIFF_SCHEMA,
                              label="разница слов")
    except Exception as e:
        info = errors.report(e, "pool_diff")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="bad diff")
    return data


@router.get("/pool/diff")
async def pool_diff(a: str, b: str, lang: str = "ru", user=Depends(get_current_user)):
    """Разница между двумя норвежскими словами на языке пользователя (LLM, кэш по паре+языку)."""
    a = normalize_word(a)
    b = normalize_word(b)
    if not a or not b or a == b:
        raise HTTPException(status_code=400, detail="two distinct words required")
    if lang not in _DIFF_LANGS:
        lang = "ru"
    ckey = f"diff|{a}|{b}|{lang}"
    cached = await get_cached_query(ckey)
    if cached:
        return {"diff": cached, "a": a, "b": b}
    data = await _gen_diff(a, b, lang)
    await cache_query(ckey, data)
    return {"diff": data, "a": a, "b": b}


@router.post("/pool/rediff")
async def pool_rediff(body: RediffBody, user=Depends(get_current_user)):
    """Перегенерировать разницу (при неверной) с учётом подсказки. Перезаписывает кэш."""
    a = normalize_word(body.a)
    b = normalize_word(body.b)
    if not a or not b or a == b:
        raise HTTPException(status_code=400, detail="two distinct words required")
    lang = body.lang if body.lang in _DIFF_LANGS else "ru"
    data = await _gen_diff(a, b, lang, (body.hint or "").strip() or None)
    await set_cached_query(f"diff|{a}|{b}|{lang}", data)
    return {"diff": data, "a": a, "b": b}


@router.get("/pool/{word}/meta")
async def pool_meta(word: str, user=Depends(get_current_user)):
    """Темы и уровень слова (для показа в карточке)."""
    meta = await get_pool_meta(word, user_id=user["id"])
    return meta or {"level": None, "topics": []}
