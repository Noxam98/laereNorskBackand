import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import logger, CORS_ORIGINS
from db import init_db, get_pool_embeddings_raw, set_pool_embedding
from llm import decode_emb, encode_emb, LLM_API_KEY
from auth import SECRET_KEY, router as auth_router
from routers.words import router as words_router
from routers.pool import router as pool_router
from autofill import (
    autofill_loop, describe_loop, translate_loop, AUTOFILL_ENABLED, AUTOFILL_DAILY_BUDGET, AUTOFILL_INTERVAL_SEC,
)

app = FastAPI()

allow_origins = ["*"] if CORS_ORIGINS.strip() == "*" else [o.strip() for o in CORS_ORIGINS.split(",")]
app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(auth_router)
app.include_router(words_router)
app.include_router(pool_router)


@app.on_event("startup")
async def startup():
    await init_db()
    if SECRET_KEY == "your_secret_key":
        logger.warning("SECRET_KEY не задан через окружение — используется значение по умолчанию.")
    # миграция эмбеддингов: legacy JSON -> бинарь float16
    try:
        migrated = 0
        for pid, raw in await get_pool_embeddings_raw():
            is_legacy = isinstance(raw, str) or (isinstance(raw, (bytes, bytearray)) and raw[:1] == b"[")
            if is_legacy:
                v = decode_emb(raw)
                if v is not None:
                    await set_pool_embedding(pid, encode_emb(v)); migrated += 1
        if migrated:
            logger.info(f"migrated {migrated} embeddings to binary float16")
    except Exception as e:
        logger.warning(f"embedding migration: {e}")
    if AUTOFILL_ENABLED:
        asyncio.create_task(autofill_loop())
        logger.info(f"autofill enabled: budget={AUTOFILL_DAILY_BUDGET}/day, interval={AUTOFILL_INTERVAL_SEC}s")
    # Очередь описаний — независимо от autofill: новые слова должны быстро получать описание.
    if LLM_API_KEY:
        asyncio.create_task(describe_loop())
        logger.info("describe queue enabled: новые слова получают описание пачками")
        asyncio.create_task(translate_loop())
        logger.info("translate queue enabled: догенерация недостающих переводов")
    # Очередь озвучки переводов в Tigris (если хранилище настроено).
    import storage
    from tts import tts_translation_loop
    if storage.enabled():
        asyncio.create_task(tts_translation_loop())
        logger.info("tts translation queue enabled: озвучка переводов → Tigris")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
