import os
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, List
from db import (
    init_db, get_user, create_user,
    get_or_create_pool, get_pool_by_id, set_pool_description,
    get_cached_query, cache_query, normalize_word,
    create_dictionary, delete_dictionary, add_word_to_dict, delete_dict_word,
    set_word_override, record_result, get_dict_word, get_user_data, search_pool,
    set_pool_embedding, get_pool_candidates,
)
import math
import random
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
        await set_pool_embedding(pool_id, vec)

def cosine(a, b):
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0

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
    data = await ask_model(task, f"Текст запроса от пользователя: >>{prompt}<<", model)
    if data is None:
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    if isinstance(data, dict):
        data = [data]
    normalized = [normalize_word_item(i) for i in data]
    await cache_query(prompt, normalized)
    return normalized, False


@app.on_event("startup")
async def startup():
    await init_db()
    if SECRET_KEY == "your_secret_key":
        logger.warning("SECRET_KEY не задан через окружение — используется значение по умолчанию.")


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
    normalized, cached = await generate_words(body.prompt, body.model)
    added, errors = 0, []
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
    return {"added": added, "errors": errors, "cached": cached}

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
    desc = await ask_model(description_task, f"Слово на норвежском: >>{dw['norwegian']}<<", model)
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
    target_emb = json.loads(dw["embedding"]) if dw["embedding"] else None

    def answer_of(data, norwegian):
        if mode == "int2no":
            return norwegian
        tr = (data.get("translate", {}) or {}).get(lang) or []
        return tr[0] if tr else None

    correct = (dw["norwegian"] if mode == "int2no" else answer_of(target, dw["norwegian"]))
    correct_l = (correct or "").strip().lower()

    cands = [c for c in await get_pool_candidates() if c["norwegian"] != dw["norwegian"]]

    if target_emb and any(c.get("embedding") for c in cands):
        scored = [(cosine(target_emb, c["embedding"]), c) for c in cands if c.get("embedding")]
        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [c for _, c in scored]
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


@app.get("/words/{dw_id}/synonyms")
async def synonyms(dw_id: int, n: int = 5, lang: str = "ru", user=Depends(get_current_user)):
    """Близкие по смыслу слова из общего пула (по эмбеддингам). Без ключа эмбеддингов — пусто."""
    dw = await get_dict_word(user["id"], dw_id)
    if not dw:
        raise HTTPException(status_code=404, detail="Not found")
    target_emb = json.loads(dw["embedding"]) if dw["embedding"] else None
    if not target_emb:
        return {"synonyms": []}
    target = json.loads(dw["data"]) if dw["data"] else {}
    target_pos = target.get("part_of_speech", "")
    cands = [c for c in await get_pool_candidates() if c["norwegian"] != dw["norwegian"] and c.get("embedding")]
    scored = [(cosine(target_emb, c["embedding"]), c) for c in cands]
    # сначала та же часть речи
    scored.sort(key=lambda x: (c2pos(x[1]) == target_pos, x[0]), reverse=True)
    out = []
    for _, c in scored[:n]:
        tr = (c["data"].get("translate", {}) or {}).get(lang) or []
        out.append({"word": c["norwegian"], "translate": tr})
    return {"synonyms": out}


def c2pos(c):
    return c["data"].get("part_of_speech", "")


# --- Shared pool (для будущего автокомплита) ---
@app.get("/pool/search")
async def pool_search(q: str, limit: int = 10, user=Depends(get_current_user)):
    return {"results": await search_pool(q, limit)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
