import os
import time
import base64
import struct
import httpx
from fastapi import FastAPI, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, List
from db import (
    init_db, get_user, create_user,
    get_or_create_pool, get_pool_by_id, get_pool_id, set_pool_description,
    get_cached_query, cache_query, normalize_word,
    create_dictionary, delete_dictionary, add_word_to_dict, delete_dict_word,
    set_word_override, record_result, get_dict_word, get_user_data, search_pool,
    set_pool_embedding, get_pool_candidates,
    get_usage, incr_usage, delete_pool_word, get_pool_list,
    get_pool_tts, set_pool_tts, pool_missing_embedding, pool_missing_tts,
    get_pool_embeddings_raw,
)
import math
import random
import asyncio
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import logging
from logging.handlers import RotatingFileHandler
from task import task, description_task

app = FastAPI()

_cors = os.getenv("CORS_ORIGINS", "*")
allow_origins = ["*"] if _cors.strip() == "*" else [o.strip() for o in _cors.split(",")]
app.add_middleware(CORSMiddleware, allow_origins=allow_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Активность = последний ПОЛЬЗОВАТЕЛЬСКИЙ вызов Google (Gemini). Фон ждёт простоя по ней.
_last_activity = time.monotonic()
_tts_lock = asyncio.Lock()

def mark_activity():
    global _last_activity
    _last_activity = time.monotonic()

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# --- LLM-провайдер (OpenAI-совместимый): Groq / OpenRouter / Ollama / любой другой через env ---
# LLM_BASE_URL, LLM_API_KEY, LLM_MODEL задаются окружением.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1:free")

_llm_client = None
def get_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI
        _llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "not-needed")
    return _llm_client

# --- Эмбеддинги (Gemini по умолчанию, OpenAI-совместимый эндпоинт; провайдер через env) ---
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")

_embed_client = None
def get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import AsyncOpenAI
        _embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "not-needed")
    return _embed_client

async def embed_text(text):
    if not EMBED_API_KEY:
        return None
    try:
        r = await get_embed_client().embeddings.create(model=EMBED_MODEL, input=text)
        await incr_usage(datetime.utcnow().strftime("%Y-%m-%d"))  # эмбеддинг — тоже вызов Gemini
        return list(r.data[0].embedding)
    except Exception as e:
        logger.warning(f"embed failed: {e}")
        return None

async def ensure_embedding(pool_id, norwegian):
    """Best-effort: посчитать и сохранить эмбеддинг слова, если его ещё нет и есть ключ."""
    if not EMBED_API_KEY:
        return
    p = await get_pool_by_id(pool_id)
    if p and p.get("embedding"):
        return
    vec = await embed_text(norwegian)
    if vec:
        await set_pool_embedding(pool_id, encode_emb(vec))

# --- TTS через Microsoft Edge (нейро-голоса, бесплатно, без ключа и без лимитов) ---
TTS_VOICE = os.getenv("TTS_VOICE", "nb-NO-FinnNeural")  # норвежский нейро-голос; ещё: nb-NO-PernilleNeural
TTS_RATE = os.getenv("TTS_RATE", "-5%")    # скорость, напр. "-15%", "+10%"
TTS_PITCH = os.getenv("TTS_PITCH", "+0Hz")  # высота, напр. "+5Hz", "-5Hz"

async def synth_tts(text: str):
    """MP3-аудио норвежского слова через Microsoft Edge TTS."""
    import edge_tts
    comm = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
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


# --- Эмбеддинги: бинарное хранение (float16) + матричный косинус (мало RAM/CPU) ---
def encode_emb(vec):
    return np.asarray(vec, dtype=np.float16).tobytes()

def decode_emb(v):
    if not v:
        return None
    if isinstance(v, (bytes, bytearray)) and not v[:1] == b"[":
        return np.frombuffer(v, dtype=np.float16).astype(np.float32)
    try:
        return np.asarray(json.loads(v), dtype=np.float32)  # legacy JSON
    except Exception:
        return None

def rank_by_similarity(target_raw, cands):
    """Вернуть cands (с эмбеддингом) по убыванию близости к target. None — у target нет вектора."""
    tv = decode_emb(target_raw)
    if tv is None:
        return None
    rows, vecs = [], []
    for c in cands:
        ev = decode_emb(c.get("embedding"))
        if ev is not None and ev.shape == tv.shape:
            rows.append(c); vecs.append(ev)
    if not rows:
        return []
    M = np.vstack(vecs)
    sims = (M @ tv) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(tv) + 1e-9) + 1e-9)
    return [rows[i] for i in np.argsort(-sims)]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_fh = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=3); _fh.setFormatter(_fmt); logger.addHandler(_fh)
_sh = logging.StreamHandler(); _sh.setFormatter(_fmt); logger.addHandler(_sh)


# --- Models ---
class UserAuth(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class RefreshRequest(BaseModel):
    refresh_token: str

class DictCreate(BaseModel):
    name: str

class AddWords(BaseModel):
    prompt: str
    model: Optional[str] = None

class ImportDict(BaseModel):
    name: str
    words: List[dict] = []

class PoolAdd(BaseModel):
    norwegian: str

class WordOverride(BaseModel):
    translate: Optional[dict] = None
    part_of_speech: Optional[str] = None

class ResultBody(BaseModel):
    correct: bool


# --- Helpers ---
def hash_password(p):
    return bcrypt.hashpw(p.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")

def verify_password(p, h):
    try:
        return bcrypt.checkpw(p.encode("utf-8")[:72], h.encode("utf-8"))
    except Exception:
        return False

def create_token(data: dict, expires: timedelta):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + expires})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user

def extract_json(content: str):
    if not content:
        return None
    m = re.search(r'```json\s*\n(.*?)\n```', content, re.DOTALL)
    candidate = m.group(1) if m else None
    if candidate is None:
        arr = re.search(r'(\[.*\])', content, re.DOTALL)
        obj = re.search(r'(\{.*\})', content, re.DOTALL)
        found = arr or obj
        candidate = found.group(1) if found else None
    if candidate is None:
        return None
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None

async def ask_model(system_prompt, user_prompt, model=None):
    resp = await get_client().chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    if resp.choices and resp.choices[0].message.content:
        return extract_json(resp.choices[0].message.content)
    return None


# Схемы для гарантированного формата ответа (structured output / JSON-schema).
_STR_ARR = {"type": "array", "items": {"type": "string"}}
WORDS_SCHEMA = {
    "name": "words_response",
    "schema": {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string", "description": "норвежское слово, без артикля, нейтральная форма"},
                        "translate": {
                            "type": "object",
                            "properties": {"ru": _STR_ARR, "ukr": _STR_ARR, "en": _STR_ARR, "pl": _STR_ARR, "lt": _STR_ARR},
                        },
                        "part_of_speech": {"type": "string"},
                    },
                    "required": ["word", "translate", "part_of_speech"],
                },
            }
        },
        "required": ["words"],
    },
}
DESC_SCHEMA = {
    "name": "description_response",
    "schema": {
        "type": "object",
        "properties": {"ru": {"type": "string"}, "ukr": {"type": "string"}, "en": {"type": "string"}, "pl": {"type": "string"}, "lt": {"type": "string"}},
        "required": ["ru", "ukr", "en", "pl", "lt"],
    },
}


async def ask_json(system_prompt, user_prompt, schema, model=None):
    """Запрос с гарантированным JSON по схеме (structured output). Фолбэк — извлечение из текста."""
    try:
        resp = await get_client().chat.completions.create(
            model=model or LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_schema", "json_schema": schema},
        )
    except Exception as e:
        logger.warning(f"structured output unsupported, fallback to plain: {e}")
        resp = await get_client().chat.completions.create(
            model=model or LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
    content = resp.choices[0].message.content if resp.choices else None
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return extract_json(content)

def normalize_word_item(item):
    if not isinstance(item, dict) or item.get("error"):
        return item
    w = item.get("word")
    if w:
        tr = item.setdefault("translate", {})
        if not tr.get("no"):
            tr["no"] = [w]
    return item

async def generate_words(prompt, model):
    """Возвращает нормализованный список слов (из кэша или AI)."""
    cached = await get_cached_query(prompt)
    if cached is not None:
        return cached, True
    if not LLM_API_KEY:
        raise HTTPException(status_code=503, detail="LLM is not configured (set LLM_API_KEY)")
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA, model)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=429, detail="Translation provider quota exhausted")
        logger.error(f"generation failed: {msg}")
        raise HTTPException(status_code=502, detail="Translation provider error")
    if data is None:
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    # гарантированный формат: { words: [...] }
    await incr_usage(datetime.utcnow().strftime("%Y-%m-%d"))  # учёт реального обращения к LLM
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    normalized = [normalize_word_item(i) for i in items]
    await cache_query(prompt, normalized)
    return normalized, False


async def persist_pool(normalized):
    """Сохранить слова в общий пул + посчитать эмбеддинги (для авто-заполнения)."""
    for item in normalized:
        if isinstance(item, dict) and not item.get("error") and item.get("word"):
            pid = await get_or_create_pool(item["word"], item)
            if pid:
                await ensure_embedding(pid, item["word"])


# --- Авто-заполнение общего пула в рамках суточного бюджета ("свободная" квота) ---
AUTOFILL_ENABLED = os.getenv("AUTOFILL_ENABLED", "false").lower() == "true"
AUTOFILL_DAILY_BUDGET = int(os.getenv("AUTOFILL_DAILY_BUDGET", "150"))
AUTOFILL_INTERVAL_SEC = int(os.getenv("AUTOFILL_INTERVAL_SEC", "300"))  # одна операция раз в N сек
AUTOFILL_BATCH = int(os.getenv("AUTOFILL_BATCH", "1"))
AUTOFILL_IDLE_SEC = int(os.getenv("AUTOFILL_IDLE_SEC", "300"))  # фон только после N сек простоя
# Ночной режим — агрессивнее (когда никто не пользуется)
AUTOFILL_NIGHT_INTERVAL_SEC = int(os.getenv("AUTOFILL_NIGHT_INTERVAL_SEC", "6"))   # ~10 операций/мин
AUTOFILL_NIGHT_BUDGET = int(os.getenv("AUTOFILL_NIGHT_BUDGET", "200"))             # потолок за день (оставляет запас квоты на день)
AUTOFILL_NIGHT_START = int(os.getenv("AUTOFILL_NIGHT_START", "2"))    # локальный час начала ночи
AUTOFILL_NIGHT_END = int(os.getenv("AUTOFILL_NIGHT_END", "7"))        # локальный час конца ночи
AUTOFILL_TZ_OFFSET = int(os.getenv("AUTOFILL_TZ_OFFSET", "2"))        # сдвиг от UTC (Норвегия летом = +2)

def _is_night():
    h = (datetime.utcnow().hour + AUTOFILL_TZ_OFFSET) % 24
    s, e = AUTOFILL_NIGHT_START, AUTOFILL_NIGHT_END
    return (s <= h < e) if s <= e else (h >= s or h < e)

AUTOFILL_TOPICS = [
    "семья", "еда", "дом", "одежда", "город", "транспорт", "погода", "работа",
    "школа", "тело человека", "животные", "эмоции", "время и даты", "покупки",
    "путешествия", "природа", "кухня и посуда", "спорт", "числа", "цвета",
    "профессии", "здоровье", "технологии", "музыка", "праздники",
]

async def autofill_loop():
    await asyncio.sleep(15)
    i = 0
    while True:
        night = _is_night()
        interval = AUTOFILL_NIGHT_INTERVAL_SEC if night else AUTOFILL_INTERVAL_SEC
        budget = AUTOFILL_NIGHT_BUDGET if night else AUTOFILL_DAILY_BUDGET
        try:
            # фон работает только в простое — после N секунд без активности юзеров
            idle = time.monotonic() - _last_activity
            if idle < AUTOFILL_IDLE_SEC:
                await asyncio.sleep(min(AUTOFILL_IDLE_SEC - idle + 1, 60))
                continue
            if LLM_API_KEY and budget > 0:
                day = datetime.utcnow().strftime("%Y-%m-%d")
                used = await get_usage(day)
                if used < budget:
                    # Приоритет: сначала добить эмбеддинги и звук у ВСЕХ слов,
                    # и только когда всё полно — генерить одно новое слово.
                    miss_e = await pool_missing_embedding(1)
                    miss_t = await pool_missing_tts(1)
                    if miss_e:
                        w = miss_e[0]
                        pid = await get_pool_id(w)
                        vec = await embed_text(w)
                        if pid and vec:
                            await set_pool_embedding(pid, encode_emb(vec))
                        logger.info(f"autofill: embedding '{w}' ok={bool(vec)}")
                    elif miss_t:
                        w = miss_t[0]
                        wav = None
                        async with _tts_lock:
                            try:
                                wav = await synth_tts(w)
                            except Exception as e:
                                logger.warning(f"autofill tts '{w}': {e}")
                            if wav:
                                await set_pool_tts(w, wav)
                        logger.info(f"autofill: tts '{w}' ok={bool(wav)}")
                    else:
                        topic = AUTOFILL_TOPICS[i % len(AUTOFILL_TOPICS)]
                        i += 1
                        normalized, _ = await generate_words(
                            f"{AUTOFILL_BATCH} распространённое норвежское слово на тему: {topic}", None
                        )
                        for it in normalized:
                            if isinstance(it, dict) and not it.get("error") and it.get("word"):
                                await get_or_create_pool(it["word"], it)  # эмбеддинг/звук добьются в след. циклах
                        logger.info(f"autofill: new word topic='{topic}'")
        except Exception as e:
            logger.warning(f"autofill error: {e}")
            await asyncio.sleep(600)  # бэкофф при ошибке/лимите
            continue
        await asyncio.sleep(interval)


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


# --- Auth ---
@app.post("/register")
async def register(user: UserAuth):
    if not user.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    if await get_user(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    return await create_user(user.username, hash_password(user.password))

@app.post("/login", response_model=Token)
async def login(creds: UserAuth):
    user = await get_user(creds.username)
    if not user or not verify_password(creds.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    sub = {"sub": creds.username}
    return {
        "access_token": create_token(sub, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
        "refresh_token": create_token(sub, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)),
        "token_type": "bearer",
    }

@app.post("/refresh", response_model=Token)
async def refresh_token(payload: RefreshRequest = None, refresh_token: str = None):
    token = (payload.refresh_token if payload else None) or refresh_token
    if not token:
        raise HTTPException(status_code=400, detail="refresh_token is required")
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = decoded.get("sub")
        if not await get_user(username):
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        sub = {"sub": username}
        return {
            "access_token": create_token(sub, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
            "refresh_token": create_token(sub, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)),
            "token_type": "bearer",
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.get("/me")
async def me(user=Depends(get_current_user)):
    return {"username": user["username"]}


# --- User data (server-backed) ---
@app.get("/data")
async def data(user=Depends(get_current_user)):
    return await get_user_data(user["id"])

@app.post("/dictionaries")
async def create_dict(body: DictCreate, user=Depends(get_current_user)):
    res = await create_dictionary(user["id"], body.name)
    if res.get("error"):
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.delete("/dictionaries/{dict_id}")
async def remove_dict(dict_id: int, user=Depends(get_current_user)):
    res = await delete_dictionary(user["id"], dict_id)
    if res.get("error"):
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.post("/dictionaries/{dict_id}/words")
async def add_words(dict_id: int, body: AddWords, user=Depends(get_current_user)):
    mark_activity()
    normalized, cached = await generate_words(body.prompt, body.model)
    added, errors, words = 0, [], []
    for item in normalized:
        if not isinstance(item, dict):
            continue
        if item.get("error"):
            errors.append(item["error"]); continue
        if not item.get("word"):
            continue
        pool_id = await get_or_create_pool(item["word"], item)
        if pool_id:
            await ensure_embedding(pool_id, item["word"])
            res = await add_word_to_dict(user["id"], dict_id, pool_id)
            if res.get("id") and not res.get("duplicate"):
                added += 1
            words.append(item["word"])
    schedule_tts(words)  # озвучку добавленных слов ставим в очередь сразу
    return {"added": added, "errors": errors, "cached": cached}

@app.post("/dictionaries/{dict_id}/add_pool")
async def add_pool(dict_id: int, body: PoolAdd, user=Depends(get_current_user)):
    """Добавить в словарь уже существующее в общем пуле слово (автокомплит) — без ИИ."""
    pid = await get_pool_id(body.norwegian)
    if not pid:
        raise HTTPException(status_code=404, detail="Word not in pool")
    res = await add_word_to_dict(user["id"], dict_id, pid)
    schedule_tts([body.norwegian])  # на случай если у слова ещё нет озвучки
    return {"added": 1 if (res.get("id") and not res.get("duplicate")) else 0, "duplicate": res.get("duplicate", False)}


@app.post("/dictionaries/import")
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

@app.delete("/words/{dw_id}")
async def remove_word(dw_id: int, user=Depends(get_current_user)):
    return await delete_dict_word(user["id"], dw_id)

@app.patch("/words/{dw_id}")
async def edit_word(dw_id: int, body: WordOverride, user=Depends(get_current_user)):
    override = {}
    if body.translate is not None:
        override["translate"] = body.translate
    if body.part_of_speech is not None:
        override["part_of_speech"] = body.part_of_speech
    return await set_word_override(user["id"], dw_id, override)

@app.post("/words/{dw_id}/result")
async def word_result(dw_id: int, body: ResultBody, user=Depends(get_current_user)):
    return await record_result(user["id"], dw_id, body.correct)

@app.get("/words/{dw_id}/description")
async def word_description(dw_id: int, model: str = None, user=Depends(get_current_user)):
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    if dw["description"]:
        return {"description": json.loads(dw["description"])}
    desc = await ask_json(description_task, f"Слово на норвежском: >>{dw['norwegian']}<<", DESC_SCHEMA, model)
    if not isinstance(desc, dict):
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    description = desc.get("description", desc)
    await set_pool_description(dw["pool_id"], description)
    return {"description": description}


@app.get("/words/{dw_id}/distractors")
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

    cands = [c for c in await get_pool_candidates() if c["norwegian"] != dw["norwegian"]]

    ranked = rank_by_similarity(dw["embedding"], cands)
    if ranked:
        ordered = ranked
    else:
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


@app.post("/words/{dw_id}/report")
async def report_word(dw_id: int, user=Depends(get_current_user)):
    """Пометить слово как неправильное: удалить из общего пула (у всех) и перегенерировать заново."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    norwegian = dw["norwegian"]
    dict_id = dw["dict_id"]
    await delete_pool_word(norwegian)  # убираем из пула у всех + чистим кэш

    regenerated, new_word = False, None
    if LLM_API_KEY:
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


@app.get("/words/{dw_id}/synonyms")
async def synonyms(dw_id: int, n: int = 5, lang: str = "ru", user=Depends(get_current_user)):
    """Близкие по смыслу слова из общего пула (по эмбеддингам). Без ключа эмбеддингов — пусто."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    if not dw["embedding"]:
        return {"synonyms": []}
    target = json.loads(dw["data"]) if dw["data"] else {}
    target_pos = target.get("part_of_speech", "")
    cands = [c for c in await get_pool_candidates() if c["norwegian"] != dw["norwegian"] and c.get("embedding")]
    ranked = rank_by_similarity(dw["embedding"], cands)
    if not ranked:
        return {"synonyms": []}
    # та же часть речи — выше (порядок по близости сохраняется внутри групп)
    ordered = [c for c in ranked if c2pos(c) == target_pos] + [c for c in ranked if c2pos(c) != target_pos]
    out = []
    for c in ordered[:n]:
        tr = (c["data"].get("translate", {}) or {}).get(lang) or []
        out.append({"word": c["norwegian"], "translate": tr})
    return {"synonyms": out}


def c2pos(c):
    return c["data"].get("part_of_speech", "")


@app.get("/tts")
async def tts(word: str):
    """Аудио произношения норвежского слова (Gemini TTS, кэшируется в пуле). Публичный."""
    key = normalize_word(word)
    if not key:
        raise HTTPException(status_code=400, detail="word is required")
    cached = await get_pool_tts(key)
    if cached:
        return Response(content=bytes(cached), media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=604800"})

    async with _tts_lock:
        cached = await get_pool_tts(key)  # могли сгенерить, пока ждали очередь
        if cached:
            return Response(content=bytes(cached), media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=604800"})
        try:
            mp3 = await synth_tts(key)
        except Exception as e:
            logger.warning(f"tts failed: {e}")
            raise HTTPException(status_code=502, detail="TTS provider error")
        if not mp3:
            raise HTTPException(status_code=502, detail="No audio")
        if await get_pool_id(key):
            await set_pool_tts(key, mp3)
        return Response(content=mp3, media_type="audio/mpeg", headers={"Cache-Control": "public, max-age=604800"})


# --- Shared pool ---
@app.get("/pool")
async def pool(q: str = None, limit: int = 60, offset: int = 0, user=Depends(get_current_user)):
    return await get_pool_list(limit, offset, q)


@app.get("/pool/search")
async def pool_search(q: str, limit: int = 10, user=Depends(get_current_user)):
    return {"results": await search_pool(q, limit)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
