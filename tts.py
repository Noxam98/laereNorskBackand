import os
import asyncio
from config import logger
from db import normalize_word, get_pool_tts, set_pool_tts, get_pool_id, tr_tts_pending, mark_tr_tts_done
from langs import TTS_VOICE_MAP, TTS_TR_PAIRS
import storage

# --- TTS через Microsoft Edge (нейро-голоса, бесплатно, без ключа и без лимитов) ---
TTS_VOICE = os.getenv("TTS_VOICE", "nb-NO-FinnNeural")  # норвежский нейро-голос; ещё: nb-NO-PernilleNeural
TTS_RATE = os.getenv("TTS_RATE", "-5%")    # скорость, напр. "-15%", "+10%"
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")  # высота, напр. "+5Hz", "-5Hz"

# Голоса по языку перевода — из реестра langs.py (добавление языка = одна запись там).
# Каждый переопределяем env-ом TTS_VOICE_<TTS> (напр. TTS_VOICE_RU). "nb" — целевой норвежский.
TTS_VOICES = {"nb": TTS_VOICE}
for _tts, _voice in TTS_VOICE_MAP.items():
    TTS_VOICES[_tts] = os.getenv(f"TTS_VOICE_{_tts.upper()}", _voice)

# Сериализует генерацию озвучки (одна за раз).
_tts_lock = asyncio.Lock()


async def synth_tts(text: str, lang: str = "nb"):
    """MP3-аудио слова через Microsoft Edge TTS. lang выбирает голос (по умолч. норвежский)."""
    import edge_tts
    voice = TTS_VOICES.get(lang, TTS_VOICE)
    comm = edge_tts.Communicate(text, voice, rate=TTS_RATE, pitch=TTS_PITCH)
    audio = bytearray()
    async for ch in comm.stream():
        if ch.get("type") == "audio" and ch.get("data"):
            audio += ch["data"]
    return bytes(audio) if audio else None


async def ensure_tts_bg(norwegian: str):
    """Сгенерировать озвучку слова в фоне (через общую очередь), если её ещё нет."""
    key = normalize_word(norwegian)
    if not key:
        return
    async with _tts_lock:
        if await get_pool_tts(key):
            return
        try:
            mp3 = await synth_tts(key)
        except Exception as e:
            logger.warning(f"bg tts '{key}': {e}")
            return
        if mp3 and await get_pool_id(key):
            await set_pool_tts(key, mp3)
            logger.info(f"bg tts ready: {key}")


def schedule_tts(words):
    for w in words:
        if w:
            asyncio.create_task(ensure_tts_bg(w))


# --- Фоновая озвучка переводов: все имеющиеся переводы каждого слова → Tigris ---
# (ключ в data.translate, код голоса озвучки) — из реестра langs.py. Украинский: "ukr" → голос "uk".
TTS_TR_LANGS = TTS_TR_PAIRS
TTS_TR_BATCH = int(os.getenv("TTS_TR_BATCH", "5"))        # слов за тик
TTS_TR_CHECK_SEC = int(os.getenv("TTS_TR_CHECK_SEC", "15"))  # период проверки очереди


async def tts_translation_loop():
    """Раз в TTS_TR_CHECK_SEC берём пачку слов без озвучки переводов, генерим
    озвучку всех их переводов (ru/uk/en/pl/lt, что есть в data.translate) и
    кладём в Tigris. Уже существующие в хранилище пропускаем. Лёгкий троттлинг."""
    if not storage.enabled():
        logger.info("tts translation loop: хранилище не настроено — пропуск")
        return
    await asyncio.sleep(20)
    while True:
        try:
            batch = await tr_tts_pending(TTS_TR_BATCH)
            if not batch:
                await asyncio.sleep(60)  # очередь пуста — ждём дольше
                continue
            made = 0
            for pid, data in batch:
                tr = (data or {}).get("translate", {}) or {}
                for dkey, vlang in TTS_TR_LANGS:
                    text = ", ".join([v for v in (tr.get(dkey) or []) if v]).strip()
                    if not text:
                        continue
                    okey = storage.key_for(vlang, text)
                    if await storage.get_object(okey):  # уже озвучено
                        continue
                    try:
                        async with _tts_lock:
                            mp3 = await synth_tts(text, vlang)
                        if mp3:
                            await storage.put_object(okey, mp3)
                            made += 1
                    except Exception as e:
                        logger.warning(f"tts_tr '{text}'[{vlang}]: {e}")
                    await asyncio.sleep(0.3)  # мягкий троттлинг edge-tts
                await mark_tr_tts_done(pid)
            if made:
                logger.info(f"tts_tr: +{made} клипов ({len(batch)} слов)")
        except Exception as e:
            logger.warning(f"tts_translation_loop: {e}")
        await asyncio.sleep(TTS_TR_CHECK_SEC)
