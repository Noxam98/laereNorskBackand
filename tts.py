import os
import asyncio
from config import logger
from db import normalize_word, get_pool_tts, set_pool_tts, get_pool_id

# --- TTS через Microsoft Edge (нейро-голоса, бесплатно, без ключа и без лимитов) ---
TTS_VOICE = os.getenv("TTS_VOICE", "nb-NO-FinnNeural")  # норвежский нейро-голос; ещё: nb-NO-PernilleNeural
TTS_RATE = os.getenv("TTS_RATE", "-5%")    # скорость, напр. "-15%", "+10%"
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")  # высота, напр. "+5Hz", "-5Hz"

# Голоса по языку перевода (нейро-голоса Edge, бесплатные). Коды как на фронте.
TTS_VOICES = {
    "nb": TTS_VOICE,
    "ru": os.getenv("TTS_VOICE_RU", "ru-RU-SvetlanaNeural"),
    "uk": os.getenv("TTS_VOICE_UK", "uk-UA-PolinaNeural"),
    "en": os.getenv("TTS_VOICE_EN", "en-US-AriaNeural"),
    "pl": os.getenv("TTS_VOICE_PL", "pl-PL-ZofiaNeural"),
    "lt": os.getenv("TTS_VOICE_LT", "lt-LT-OnaNeural"),
}

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
