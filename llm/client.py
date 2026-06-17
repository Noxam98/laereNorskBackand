"""Транспорт к провайдеру: OpenAI-совместимые клиенты + примитивы запроса
(ask_json / ask_model / embed_text / embed_texts). Ключи берёт у quota.py по «дневнику
429»: свежие ключи впереди, на 429 помечаем ключ и СРАЗУ повторяем следующим. Наружу
ключей не видно."""
import json
import re
from config import logger
import errors
import notify
from .settings import LLM_BASE_URL, EMBED_BASE_URL, LLM_API_KEY, EMBED_API_KEY, LLM_MODEL
from . import quota

_llm_client = None
_embed_client = None


def get_client():
    global _llm_client
    if _llm_client is None:
        from openai import AsyncOpenAI
        _llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY or "not-needed")
    return _llm_client


def get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import AsyncOpenAI
        # max_retries=0 — не висим минутами на внутренних ретраях клиента при 429;
        # повторами/паузами управляем сами. timeout — чтобы не зависать.
        _embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "not-needed",
                                    max_retries=0, timeout=30)
    return _embed_client


async def _run(kind, cands, attempt, incr, label, icon):
    """Перебрать кандидатов [(model, key, idx)]; attempt(model, key) -> результат.
    На 429: пометить ключ в дневнике (скип на COOLDOWN_SEC) и СРАЗУ пробовать следующего.
    Не-429 ошибка пробрасывается сразу. Успех — запоминаем ключ (round-robin) и учитываем
    расход. Возвращает результат, либо бросает последнюю 429, если 429 на всех ключах."""
    if not cands:
        notify.feed(f"{icon} {label} ⛔ ключей нет")
        return None
    last = None
    for model, key, idx in cands:
        try:
            res = await attempt(model, key)
        except Exception as e:
            ek = errors.classify(e).kind
            if ek != errors.QUOTA:
                notify.feed(f"{icon} {label} [{model}] · k{idx} ❌ {ek}")
                raise
            quota.mark_429(kind, model, idx)  # скипаем этот ключ COOLDOWN_SEC секунд
            notify.feed(f"{icon} {label} [{model}] · k{idx} ⚠️429 → следующий ключ")
            last = e
            continue
        quota.advance(kind, model, idx)        # next-запрос начнём со следующего ключа
        await incr(model, key)                 # учёт для статистики
        notify.feed(f"{icon} {label} [{model}] · k{idx} ✅")
        return res
    notify.feed(f"{icon} {label} ⛔ 429 на всех ключах")
    if last:
        raise last
    return None


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


async def ask_model(system_prompt, user_prompt, model=None, api_key=None):
    client = get_client().with_options(api_key=api_key or LLM_API_KEY or "not-needed")
    resp = await client.chat.completions.create(
        model=model or LLM_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    if resp.choices and resp.choices[0].message.content:
        return extract_json(resp.choices[0].message.content)
    return None


async def ask_json(system_prompt, user_prompt, schema, purpose="user", label="LLM-запрос", model=None):
    """Запрос с гарантированным JSON по схеме (structured output). Фолбэк — извлечение из текста.
    purpose — профиль ("user" | "autofill"); model — необязательный override. Ключ/429 — внутри."""
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def attempt(m, key):
        client = get_client().with_options(api_key=key or "not-needed")
        try:
            resp = await client.chat.completions.create(
                model=m or LLM_MODEL, messages=msgs,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        except Exception as e:
            # Фолбэк на простой запрос — ТОЛЬКО если провайдер не понял схему (400).
            if errors.classify(e).kind != errors.BAD_REQUEST:
                raise
            logger.warning(f"structured output unsupported, fallback to plain: {e}")
            resp = await client.chat.completions.create(model=m or LLM_MODEL, messages=msgs)
        return resp

    resp = await _run("text", quota.text_candidates(purpose, model), attempt, quota.incr_text, label, "📝")
    content = resp.choices[0].message.content if (resp and resp.choices) else None
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return extract_json(content)


async def embed_text(text, label=None):
    if not quota.embed_enabled():
        return None

    async def attempt(m, key):
        r = await get_embed_client().with_options(api_key=key).embeddings.create(model=m, input=text)
        return list(r.data[0].embedding)

    try:
        return await _run("emb", quota.embed_candidates(), attempt, quota.incr_emb,
                          label or "эмбеддинг 1 слова", "🧮")
    except Exception as e:
        errors.report(e, "embed_text")
        return None


async def embed_texts(texts, label=None):
    """Батч-эмбеддинг списка текстов одним запросом (до 100). Возвращает список векторов
    в исходном порядке или None при ошибке/исчерпании ключей. Ключ/модель/429 — внутри."""
    if not quota.embed_enabled() or not texts:
        return None

    async def attempt(m, key):
        r = await get_embed_client().with_options(api_key=key).embeddings.create(model=m, input=texts)
        data = list(r.data)  # API возвращает в порядке входа
        if all(getattr(d, "index", None) is not None for d in data):
            data.sort(key=lambda d: d.index)  # подстраховка по индексу, если он есть
        return [list(d.embedding) for d in data]

    try:
        return await _run("emb", quota.embed_candidates(), attempt, quota.incr_emb,
                          label or f"эмбеддинг {len(texts)} слов", "🧮")
    except Exception as e:
        errors.report(e, "embed_texts")
        return None
