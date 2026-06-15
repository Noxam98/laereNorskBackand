from fastapi import APIRouter, Depends, HTTPException, Response
from config import logger
from db import (
    normalize_word, get_pool_tts, set_pool_tts, get_pool_id, get_pool_list,
    search_pool, get_pool_topics_counts, get_pool_level_counts,
)
from auth import get_current_user
from tts import synth_tts, _tts_lock
from llm import TOPIC_KEYS, CEFR_LEVELS

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


@router.get("/pool/search")
async def pool_search(q: str, limit: int = 10, user=Depends(get_current_user)):
    return {"results": await search_pool(q, limit)}
