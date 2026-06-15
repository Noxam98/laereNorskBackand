from fastapi import APIRouter, Depends, HTTPException, Response
from config import logger
from datetime import datetime
from db import (
    normalize_word, get_pool_tts, set_pool_tts, get_pool_id, get_pool_by_id,
    set_pool_description, get_pool_list,
    search_pool, get_pool_topics_counts, get_pool_level_counts, get_pool_stats, get_usage_like,
)
from auth import get_current_user, get_admin_user
from activity import mark_activity
from tts import synth_tts, _tts_lock
from llm import TOPIC_KEYS, CEFR_LEVELS, ask_json, DESC_SCHEMA, ranked_pool
from task import description_task

router = APIRouter()

_TTS_HEADERS = {"Cache-Control": "public, max-age=604800"}


@router.get("/tts")
async def tts(word: str):
    """Аудио произношения норвежского слова (Edge TTS, кэшируется в пуле). Публичный."""
    key = normalize_word(word)
    if not key:
        raise HTTPException(status_code=400, detail="word is required")
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
    return await get_pool_list(limit, offset, q, topic_list, lvl, srt, order)


@router.get("/pool/topics")
async def pool_topics(user=Depends(get_current_user)):
    return {"topics": await get_pool_topics_counts(), "levels": await get_pool_level_counts()}


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
