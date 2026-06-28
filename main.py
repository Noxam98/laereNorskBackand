import os
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import logger, CORS_ORIGINS
from db import init_db, get_pool_embeddings_raw, set_pool_embedding
from llm import decode_emb, encode_emb, text_enabled
from auth import router as auth_router
from routers.words import router as words_router
from routers.pool import router as pool_router
from online import router as online_router
from routers.learning import router as learning_router
from routers.sets import router as sets_router
from autofill import (
    autofill_loop, describe_loop, translate_loop, reembed_loop, forms_loop, pos_loop, dedup_loop, freq_loop,
    yo_fix_loop,
    AUTOFILL_ENABLED, AUTOFILL_DAILY_BUDGET, AUTOFILL_INTERVAL_SEC, DEDUP_ENABLED,
)

# Swagger/OpenAPI скрываем в проде (ENABLE_DOCS=1 чтобы включить) — не светим карту эндпоинтов.
_DOCS = os.getenv("ENABLE_DOCS", "0") == "1"
app = FastAPI(docs_url="/docs" if _DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if _DOCS else None)

# CORS: НИКОГДА не пара wildcard+credentials (невалидно по спеку и опасно). Конкретный список
# origin → с креденшелами; "*" → без них. Фронт ходит Bearer-токеном (не cookie) — креды не нужны.
_wild = CORS_ORIGINS.strip() == "*"
allow_origins = ["*"] if _wild else [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=not _wild,
                   allow_methods=["*"], allow_headers=["*"])


# Базовые security-заголовки на все ответы. TLS форсится на edge (fly.toml force_https).
@app.middleware("http")
async def _security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return resp

app.include_router(auth_router)
app.include_router(words_router)
app.include_router(pool_router)
app.include_router(online_router)
app.include_router(learning_router)
app.include_router(sets_router)
# Веб-пуши — подключаем защищённо: если модуль/зависимость отвалятся, бэкенд всё равно поднимется.
try:
    from webpush import router as push_router
    app.include_router(push_router)
except Exception as _e:
    logger.warning(f"web push router not loaded: {_e}")


@app.on_event("startup")
async def startup():
    await init_db()
    import runtime
    await runtime.load_persisted()  # восстановить паузы фоновых задач из БД
    import notify
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
    # чистка форм у частей речи, которым они не положены (мусор от прошлой коллизии noun/pronoun, verb/adverb)
    try:
        from db import clear_nonformable_forms
        cleared = await clear_nonformable_forms()
        if cleared:
            logger.info(f"cleared grammatical forms from {cleared} non-formable words")
    except Exception as e:
        logger.warning(f"forms cleanup: {e}")
    # Разовый сброс всех описаний (фон догенерит заново под новый промпт — с частью речи и
    # переводами). Гард по версии-настройке: выполняется один раз на это значение.
    try:
        from db import get_setting, set_setting, clear_all_descriptions
        DESC_RESET_VER = "v2_pos_translate"
        if await get_setting("desc_reset") != DESC_RESET_VER:
            await clear_all_descriptions()
            await set_setting("desc_reset", DESC_RESET_VER)
            logger.info("all word descriptions reset — will be regenerated with richer prompt")
    except Exception as e:
        logger.warning(f"desc reset: {e}")
    if text_enabled():
        if AUTOFILL_ENABLED:
            asyncio.create_task(autofill_loop())
            logger.info(f"autofill loop started: budget={AUTOFILL_DAILY_BUDGET}/day, interval={AUTOFILL_INTERVAL_SEC}s")
        asyncio.create_task(describe_loop())
        logger.info("describe queue enabled: новые слова получают описание пачками")
        asyncio.create_task(translate_loop())
        logger.info("translate queue enabled: догенерация недостающих переводов")
        asyncio.create_task(reembed_loop())
        logger.info("reembed queue enabled: пере-эмбеддинг пула по смыслу")
        asyncio.create_task(pos_loop())
        logger.info("pos queue enabled: переразметка части речи у «прочее»")
        asyncio.create_task(forms_loop())
        logger.info("forms queue enabled: грамматические формы по части речи")
        if DEDUP_ENABLED:
            asyncio.create_task(dedup_loop())
            logger.info("dedup queue enabled: фоновое слияние слов-дублей пула")
        asyncio.create_task(freq_loop())
        logger.info("freq queue enabled: простановка частотности слов (Zipf)")
        asyncio.create_task(yo_fix_loop())
        logger.info("yo-fix queue enabled: бэкилл буквы «ё» в русских переводах + переозвучка")
    # Telegram — только оповещения: алерты (notify) + лента активности «что происходит».
    asyncio.create_task(notify.feed_worker())
    logger.info(f"telegram notifications: {'ON' if notify.enabled() else 'OFF'} · feed: {'ON' if notify.FEED_ON else 'OFF'}")
    # Веб-пуши: напоминание «13ч бездействия». No-op, если VAPID-ключи не заданы. Защищённо.
    try:
        from webpush import reminder_loop as push_reminder_loop, configured as push_configured
        asyncio.create_task(push_reminder_loop())
        logger.info(f"web push reminders: {'ON' if push_configured() else 'OFF (no VAPID)'}")
    except Exception as e:
        logger.warning(f"web push reminders not started: {e}")
    # Очередь озвучки переводов в Tigris (если хранилище настроено).
    import storage
    from tts import tts_translation_loop
    if storage.enabled():
        asyncio.create_task(tts_translation_loop())
        logger.info("tts translation queue enabled: озвучка переводов → Tigris")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
