import os
import re
import json
import asyncio
from datetime import datetime
import numpy as np
from fastapi import HTTPException
from config import logger
import errors
from db import (
    get_cached_query, cache_query, incr_usage, get_usage,
    get_pool_by_id, set_pool_embedding, get_or_create_pool, set_pool_meta,
    vec_nearest_rows, get_pool_candidates,
)
from task import task

# --- LLM-провайдер (OpenAI-совместимый): Gemini / Groq / OpenRouter / любой через env ---
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
        # max_retries=0 — не висим минутами на внутренних ретраях клиента при 429;
        # пейсингом и повторами управляет наш цикл. timeout — чтобы не зависать.
        _embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "not-needed",
                                    max_retries=0, timeout=30)
    return _embed_client


async def embed_text(text, model=None):
    if not EMBED_API_KEY:
        return None
    m = model or EMBED_MODEL
    try:
        r = await get_embed_client().embeddings.create(model=m, input=text)
        await incr_usage(datetime.utcnow().strftime("%Y-%m-%d") + ":emb:" + m)  # учёт по модели
        return list(r.data[0].embedding)
    except Exception as e:
        errors.report(e, f"embed_text({m})")
        return None


_EMB_LANGS = ["ru", "ukr", "en", "pl", "lt"]


def semantic_embed_text(data):
    """Текст для эмбеддинга по СМЫСЛУ: норвежское слово + все переводы.
    Так вектор отражает значение, а не написание (соседи — по смыслу)."""
    data = data or {}
    tr = data.get("translate", {}) or {}
    parts = [data.get("word") or (tr.get("no") or [""])[0]]
    for l in _EMB_LANGS:
        parts.extend(v for v in (tr.get(l) or []) if v)
    return ", ".join(p for p in parts if p).strip()


async def embed_texts(texts, model=None):
    """Батч-эмбеддинг списка текстов одним запросом (до 100). Возвращает список
    векторов в исходном порядке или None при ошибке. Один запрос = одна единица квоты."""
    if not EMBED_API_KEY or not texts:
        return None
    m = model or EMBED_MODEL
    try:
        r = await get_embed_client().embeddings.create(model=m, input=texts)
        await incr_usage(datetime.utcnow().strftime("%Y-%m-%d") + ":emb:" + m)
        data = list(r.data)  # API возвращает в порядке входа
        if all(getattr(d, "index", None) is not None for d in data):
            data.sort(key=lambda d: d.index)  # подстраховка по индексу, если он есть
        return [list(d.embedding) for d in data]
    except Exception as e:
        errors.report(e, f"embed_texts({m})")
        return None


async def ensure_embedding(pool_id, norwegian):
    """Best-effort: посчитать и сохранить эмбеддинг слова по смыслу, если его ещё нет."""
    if not EMBED_API_KEY:
        return
    p = await get_pool_by_id(pool_id)
    if not p or p.get("embedding"):
        return
    vec = await embed_text(semantic_embed_text(p["data"]) or norwegian)
    if vec:
        await set_pool_embedding(pool_id, encode_emb(vec))


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


async def ranked_pool(target_raw, exclude_norwegian, n):
    """Ближайшие по смыслу слова пула [{norwegian, data(dict)}], исключая exclude.
    Использует ANN-индекс (sqlite-vec) если доступен, иначе brute-force в отдельном потоке."""
    if not target_raw:
        return []
    rows = await vec_nearest_rows(target_raw, n + 5)  # None — индекс недоступен
    if rows is not None:
        out = []
        for r in rows:
            if r["norwegian"] == exclude_norwegian:
                continue
            out.append({"norwegian": r["norwegian"], "data": json.loads(r["data"]) if r["data"] else {}})
            if len(out) >= n:
                break
        return out
    # фолбэк: перебор всех кандидатов (CPU — в треде, чтобы не блокировать event loop)
    cands = [c for c in await get_pool_candidates() if c["norwegian"] != exclude_norwegian and c.get("embedding")]
    ranked = await asyncio.to_thread(rank_by_similarity, target_raw, cands)
    if not ranked:
        return []
    return [{"norwegian": c["norwegian"], "data": c["data"]} for c in ranked[:n]]


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


# Канонические темы-теги пула (стабильные ключи; UI-подписи — во фронтовом i18n).
# Значение — рус. подсказка для LLM-классификатора и генерации.
TOPIC_TAGS = {
    "family": "семья и родственники",
    "food": "еда, напитки, продукты",
    "home": "дом, мебель, быт, посуда",
    "work": "работа, профессии, офис, бизнес",
    "school": "школа, учёба, образование, наука",
    "travel": "путешествия, туризм, отдых, гостиница",
    "health": "здоровье, болезни, медицина, аптека",
    "body": "тело человека, органы чувств",
    "clothing": "одежда, обувь, аксессуары",
    "nature": "природа, ландшафт, растения, деревья",
    "animals": "животные, птицы, рыбы, насекомые",
    "weather": "погода, климат, времена года",
    "city": "город, здания, места, улицы",
    "transport": "транспорт, машины, дорога, движение",
    "shopping": "деньги, покупки, магазин, финансы",
    "time": "время, даты, дни, месяцы",
    "sport": "спорт, фитнес, игры",
    "hobby": "хобби, досуг, культура, искусство, музыка, кино",
    "technology": "технологии, компьютеры, интернет, гаджеты",
    "communication": "общение, речь, связь, приветствия",
    "emotions": "эмоции, чувства, черты характера",
    "holidays": "праздники, традиции, события",
    "society": "общество, политика, право, экономика",
    "other": "прочее, абстрактные понятия, качества, количества",
}
TOPIC_KEYS = list(TOPIC_TAGS.keys())
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

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
                        "level": {"type": "string", "enum": CEFR_LEVELS},
                        "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_KEYS}},
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
# Разница между двумя норвежскими словами (на языке пользователя).
DIFF_SCHEMA = {
    "name": "word_diff_response",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},   # суть различия одной фразой
            "when_a": {"type": "string"},     # когда употреблять первое слово
            "when_b": {"type": "string"},     # когда употреблять второе слово
            "example": {"type": "string"},    # короткий пример (норвежский + перевод)
        },
        "required": ["summary", "when_a", "when_b", "example"],
    },
}
# Пакетная классификация слов: уровень CEFR + 1-3 темы из канонического списка.
CLASSIFY_SCHEMA = {
    "name": "classify_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "level": {"type": "string", "enum": CEFR_LEVELS},
                        "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_KEYS}, "minItems": 1, "maxItems": 3},
                    },
                    "required": ["word", "level", "topics"],
                },
            }
        },
        "required": ["results"],
    },
}


DESCRIBE_BATCH_SCHEMA = {
    "name": "describe_batch_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "ru": {"type": "string"}, "ukr": {"type": "string"}, "en": {"type": "string"},
                        "pl": {"type": "string"}, "lt": {"type": "string"},
                    },
                    "required": ["word", "ru", "ukr", "en", "pl", "lt"],
                },
            }
        },
        "required": ["results"],
    },
}


# Пакетный перевод: для каждого норвежского слова — варианты перевода на 5 языков.
TRANSLATE_BATCH_SCHEMA = {
    "name": "translate_batch_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "ru": _STR_ARR, "ukr": _STR_ARR, "en": _STR_ARR, "pl": _STR_ARR, "lt": _STR_ARR,
                    },
                    "required": ["word", "ru", "ukr", "en", "pl", "lt"],
                },
            }
        },
        "required": ["results"],
    },
}


# Уточнение перевода группы слов (одинаковые/неточные переводы) на один язык.
REFINE_SCHEMA = {
    "name": "refine_translate_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}, "translate": _STR_ARR},
                    "required": ["word", "translate"],
                },
            }
        },
        "required": ["results"],
    },
}

# Названия языков перевода (ключ интерфейса → язык на русском для промпта).
LANG_NAMES = {"ru": "русский", "ukr": "украинский", "en": "английский", "pl": "польский", "lt": "литовский"}


async def refine_translations(items, lang, model=None):
    """items: [{"word": no, "current": [..]}]. Вернуть {word_lower: [переводы]} —
    точные, различимые между собой переводы на язык `lang`."""
    lang_name = LANG_NAMES.get(lang, lang)
    lines = "\n".join(f"- {it['word']}: {', '.join(it.get('current') or []) or '—'}" for it in items)
    system = (
        f"Ты эксперт-лексикограф норвежского языка. Пользователь выделил группу слов, "
        f"у которых перевод на {lang_name} получился одинаковым или слишком общим. "
        f"Для КАЖДОГО слова дай точный перевод на {lang_name}, подобранный так, чтобы слова "
        f"в группе были различимы между собой (без идентичных переводов). 1-3 самых точных "
        f"варианта на слово, сохраняя часть речи. Отвечай строго по схеме."
    )
    user = f"Слова (норвежское: текущий перевод на {lang_name}):\n{lines}"
    res = await ask_json(system, user, REFINE_SCHEMA, model)
    out = {}
    if isinstance(res, dict):
        for r in res.get("results", []):
            w = (r.get("word") or "").strip().lower()
            tr = [t.strip() for t in (r.get("translate") or []) if isinstance(t, str) and t.strip()]
            if w and tr:
                out[w] = tr
    return out


async def ask_json(system_prompt, user_prompt, schema, model=None):
    """Запрос с гарантированным JSON по схеме (structured output). Фолбэк — извлечение из текста."""
    try:
        resp = await get_client().chat.completions.create(
            model=model or LLM_MODEL,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_schema", "json_schema": schema},
        )
    except Exception as e:
        # Фолбэк на простой запрос имеет смысл ТОЛЬКО когда провайдер не понял схему (400).
        # При квоте/ключе/сбое/таймауте второй запрос бесполезен — пробрасываем, чтобы
        # вызывающий обработал ошибку правильно (классификация + уведомление).
        if errors.classify(e).kind != errors.BAD_REQUEST:
            raise
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


# Модели для ЖИВЫХ запросов пользователей: «по возможности» лучшая модель, при
# исчерпании её суточной квоты — следующая. Формат env: "model:rpd,...".
# По умолчанию — только LLM_MODEL (без ограничения), т.е. прежнее поведение.
def _parse_models(env, default_model, default_budget=10 ** 9):
    out = []
    for part in (os.getenv(env, "") or "").split(","):
        part = part.strip()
        if not part:
            continue
        m, _, b = part.rpartition(":")
        try:
            out.append((m.strip(), int(b)))
        except ValueError:
            out.append((part, default_budget))
    return out or [(default_model, default_budget)]


USER_TEXT_MODELS = _parse_models("USER_TEXT_MODELS", LLM_MODEL)


async def pick_user_text_model():
    """Лучшая доступная по суточной квоте модель для запросов пользователя."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    for m, budget in USER_TEXT_MODELS:
        if (await get_usage(f"{today}:text:{m}")) < budget:
            return m
    return USER_TEXT_MODELS[-1][0]


async def generate_words(prompt, model):
    """Возвращает нормализованный список слов (из кэша или AI)."""
    cached = await get_cached_query(prompt)
    if cached is not None:
        return cached, True
    if not LLM_API_KEY:
        raise HTTPException(status_code=503, detail="LLM is not configured (set LLM_API_KEY)")
    if not model:
        model = await pick_user_text_model()
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA, model)
    except Exception as e:
        info = errors.report(e, "generate_words")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    if data is None:
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    # гарантированный формат: { words: [...] }
    await incr_usage(datetime.utcnow().strftime("%Y-%m-%d") + ":text:" + (model or LLM_MODEL))  # учёт по модели
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    normalized = [normalize_word_item(i) for i in items]
    await cache_query(prompt, normalized)
    return normalized, False


async def apply_item_meta(pid, item):
    """Проставить уровень/темы из сгенерированного слова, если пришли валидными."""
    level = item.get("level") if item.get("level") in CEFR_LEVELS else None
    topics = [t for t in (item.get("topics") or []) if t in TOPIC_KEYS]
    if level or topics:
        await set_pool_meta(pid, level=level, topics=topics or None)


async def persist_pool(normalized):
    """Сохранить слова в общий пул + посчитать эмбеддинги (для авто-заполнения)."""
    for item in normalized:
        if isinstance(item, dict) and not item.get("error") and item.get("word"):
            pid = await get_or_create_pool(item["word"], item)
            if pid:
                await ensure_embedding(pid, item["word"])
                await apply_item_meta(pid, item)
