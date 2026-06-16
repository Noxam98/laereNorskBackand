from fastapi import APIRouter, Depends, HTTPException, Response
from config import logger
import errors
import asyncio
from datetime import datetime
from db import (
    normalize_word, get_pool_tts, set_pool_tts, get_pool_id, get_pool_by_id,
    set_pool_description, get_pool_list, delete_pool_word, pool_missing_description,
    search_pool, get_pool_topics_counts, get_pool_level_counts, get_pool_facets, get_pool_meta, get_pool_stats, get_usage_like,
    get_cached_query, cache_query, set_cached_query,
)
from auth import get_current_user, get_admin_user
from activity import mark_activity
from tts import synth_tts, _tts_lock
from llm import TOPIC_KEYS, CEFR_LEVELS, ask_json, DESC_SCHEMA, DIFF_SCHEMA, ranked_pool
from task import description_task
from models import RedescribeBody, RediffBody
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


# --- Shared pool ---
@router.get("/pool")
async def pool(q: str = None, limit: int = 60, offset: int = 0,
              topics: str = None, level: str = None,
              sort: str = "alpha", order: str = "asc", user=Depends(get_current_user)):
    topic_list = [t for t in (topics.split(",") if topics else []) if t in TOPIC_KEYS]
    lvl = level if level in CEFR_LEVELS else None
    srt = sort if sort in ("alpha", "level", "added") else "alpha"
    res = await get_pool_list(limit, offset, q, topic_list, lvl, srt, order)
    res["facets"] = await get_pool_facets(q, topic_list, lvl)  # динамические счётчики под текущий фильтр
    return res


@router.get("/pool/topics")
async def pool_topics(user=Depends(get_current_user)):
    return {"topics": await get_pool_topics_counts(), "levels": await get_pool_level_counts()}


@router.delete("/admin/pool/{word}")
async def admin_delete_word(word: str, user=Depends(get_admin_user)):
    """Полностью удалить слово из общего пула (у всех + кэш + ANN-индекс). Только админ."""
    await delete_pool_word(word)
    return {"ok": True}


@router.post("/admin/describe_all")
async def admin_describe_all(user=Depends(get_admin_user)):
    """Запустить фоновую пакетную догенерацию описаний для всех слов без описания."""
    from autofill import describe_all_task
    pending = len(await pool_missing_description(1000000))
    asyncio.create_task(describe_all_task())
    return {"pending": pending, "started": True}


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
async def pool_search(q: str, limit: int = 10, user=Depends(get_current_user)):
    return {"results": await search_pool(q, limit)}


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
    desc = await ask_json(description_task, f"Слово на норвежском: >>{normalize_word(word)}<<", DESC_SCHEMA, model)
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
    user_prompt = f"Слово на норвежском: >>{normalize_word(word)}<<"
    if hint:
        user_prompt += ("\nВАЖНО: предыдущее описание было неверным. Правильное значение/уточнение "
                        f"от пользователя (учти обязательно): {hint}")
    desc = await ask_json(description_task, user_prompt, DESC_SCHEMA)
    if not isinstance(desc, dict):
        raise HTTPException(status_code=502, detail="No JSON")
    description = desc.get("description", desc)
    await set_pool_description(pid, description)
    return {"description": description}


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
    meta = await get_pool_meta(word)
    return meta or {"level": None, "topics": []}
