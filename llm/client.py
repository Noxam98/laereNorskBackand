"""Транспорт к провайдеру: OpenAI-совместимые клиенты, failover ключей на 429 и
примитивы запроса — ask_json / ask_model / embed_text / embed_texts. Ключи берёт у
quota.py (round-robin) и ему же отдаёт учёт расхода; наружу ключей не видно."""
import json
import re
from config import logger
import errors
import notify
from .settings import (
    LLM_BASE_URL, EMBED_BASE_URL, LLM_API_KEY, EMBED_API_KEY,
    LLM_API_KEYS, EMBED_API_KEYS, LLM_MODEL,
)
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
        # пейсингом и повторами управляет наш цикл. timeout — чтобы не зависать.
        _embed_client = AsyncOpenAI(base_url=EMBED_BASE_URL, api_key=EMBED_API_KEY or "not-needed",
                                    max_retries=0, timeout=30)
    return _embed_client


async def failover(attempt, start_key, keys=None, label="LLM", icon="📝"):
    """Выполнить attempt(key); при 429/квоте — повторить следующим ключом пула (по кругу
    от start_key). Бросает последнюю 429-ошибку, если ВСЕ ключи исчерпаны, либо сразу
    любую НЕ-квотную ошибку. Пишет осмысленную строку в ленту (операция + исход)."""
    keys = keys or LLM_API_KEYS or [start_key]
    n = len(keys)
    try:
        start = keys.index(start_key)
    except (ValueError, AttributeError):
        start = 0
    last = None
    for i in range(n):
        idx = (start + i) % n
        try:
            res = await attempt(keys[idx])
            notify.feed(f"{icon} {label} · k{idx} ✅")
            return res
        except Exception as e:
            kind = errors.classify(e).kind
            if kind != errors.QUOTA:
                notify.feed(f"{icon} {label} · k{idx} ❌ {kind}")
                raise
            last = e
            if i < n - 1:
                notify.feed(f"{icon} {label} · k{idx} ⚠️429 → переключаюсь на k{(idx + 1) % n}")
                logger.warning(f"429 на k{idx} ({label}) — пробую k{(idx + 1) % n}")
            else:
                notify.feed(f"{icon} {label} · k{idx} ⛔ 429 (все ключи исчерпаны)")
    raise last


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
    purpose — профиль квоты ("user" | "autofill"); model — необязательный override.
    Ключ/учёт/429 — внутри. label — назначение запроса для ленты активности."""
    key, model = await quota.pick_text(purpose, model)
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    async def one(k):
        client = get_client().with_options(api_key=k or "not-needed")
        try:
            resp = await client.chat.completions.create(
                model=model or LLM_MODEL, messages=msgs,
                response_format={"type": "json_schema", "json_schema": schema},
            )
        except Exception as e:
            # Фолбэк на простой запрос — ТОЛЬКО если провайдер не понял схему (400).
            if errors.classify(e).kind != errors.BAD_REQUEST:
                raise
            logger.warning(f"structured output unsupported, fallback to plain: {e}")
            resp = await client.chat.completions.create(model=model or LLM_MODEL, messages=msgs)
        await quota.incr_text(model or LLM_MODEL, k)  # учёт по фактически использованному ключу
        return resp

    resp = await failover(one, key, label=f"{label} ({model or LLM_MODEL})", icon="📝")
    content = resp.choices[0].message.content if resp.choices else None
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return extract_json(content)


async def embed_text(text, label=None):
    if not EMBED_API_KEYS:
        return None
    key, m = await quota.pick_emb()

    async def one(k):
        r = await get_embed_client().with_options(api_key=k).embeddings.create(model=m, input=text)
        await quota.incr_emb(m, k)  # учёт по фактически использованному ключу
        return list(r.data[0].embedding)

    try:
        return await failover(one, key, keys=EMBED_API_KEYS,
                              label=label or f"эмбеддинг 1 слова ({m})", icon="🧮")
    except Exception as e:
        errors.report(e, f"embed_text({m})")
        return None


async def embed_texts(texts, label=None):
    """Батч-эмбеддинг списка текстов одним запросом (до 100). Возвращает список векторов
    в исходном порядке или None при ошибке/исчерпании ключей. Ключ/модель — внутри
    (round-robin + failover на 429). Один запрос = одна единица квоты."""
    if not EMBED_API_KEYS or not texts:
        return None
    key, m = await quota.pick_emb()

    async def one(k):
        r = await get_embed_client().with_options(api_key=k).embeddings.create(model=m, input=texts)
        await quota.incr_emb(m, k)
        data = list(r.data)  # API возвращает в порядке входа
        if all(getattr(d, "index", None) is not None for d in data):
            data.sort(key=lambda d: d.index)  # подстраховка по индексу, если он есть
        return [list(d.embedding) for d in data]

    try:
        return await failover(one, key, keys=EMBED_API_KEYS,
                              label=label or f"эмбеддинг {len(texts)} слов ({m})", icon="🧮")
    except Exception as e:
        errors.report(e, f"embed_texts({m})")
        return None
