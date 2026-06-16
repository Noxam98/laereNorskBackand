"""Ключи и квота: round-robin ключей внутри модели, суточные бюджеты, учёт расхода.
Единственное место, которое знает про конкретные ключи. Наружу отдаёт key-free хелперы
(text_budget_left/embed_budget_left/quota_snapshot/...). Транспорт (client.py) спрашивает
здесь «дай ключ под этот purpose» и «учти расход»."""
from datetime import datetime
from db import get_usage, incr_usage
from .settings import LLM_API_KEYS, EMBED_API_KEYS, TEXT_PROFILES, EMBED_MODELS


def _today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def _key_id(api_key, pool):
    try:
        return pool.index(api_key)
    except (ValueError, AttributeError):
        return 0


_rr_cursor = {}  # (kind, model) -> индекс последнего выданного ключа (для round-robin)


async def pick(models, keys, kind):
    """Пара (ключ, модель). Модели — в порядке приоритета; внутри модели ключи идут
    ПО КРУГУ: каждый следующий запрос к одной модели — со следующего ключа. Пропускаем
    ключи, исчерпавшие суточный бюджет; у модели исчерпаны ВСЕ ключи → следующая модель.
    (None, None) — всё исчерпано."""
    today = _today()
    n = len(keys)
    for m, budget in models:
        start = _rr_cursor.get((kind, m), -1)
        for off in range(1, n + 1):
            idx = (start + off) % n
            if (await get_usage(f"{today}:{kind}:{m}:k{idx}")) < budget:
                _rr_cursor[(kind, m)] = idx
                return keys[idx], m
    return None, None


async def incr_text(model, key):
    await incr_usage(f"{_today()}:text:{model}:k{_key_id(key, LLM_API_KEYS)}")


async def incr_emb(model, key):
    await incr_usage(f"{_today()}:emb:{model}:k{_key_id(key, EMBED_API_KEYS)}")


async def pick_text(purpose, model=None):
    """(ключ, модель) для текстового запроса profile. model — явный override."""
    if model:
        key, _ = await pick([(model, 10 ** 9)], LLM_API_KEYS, "text")
        return (key or (LLM_API_KEYS[0] if LLM_API_KEYS else None)), model
    key, m = await pick(TEXT_PROFILES[purpose], LLM_API_KEYS, "text")
    if not m:  # всё исчерпано — деградируем на последнюю модель + первый ключ
        return (LLM_API_KEYS[0] if LLM_API_KEYS else None), TEXT_PROFILES[purpose][-1][0]
    return key, m


async def pick_emb():
    key, m = await pick(EMBED_MODELS, EMBED_API_KEYS, "emb")
    if not m:
        return (EMBED_API_KEYS[0] if EMBED_API_KEYS else None), EMBED_MODELS[-1][0]
    return key, m


# --- Публичный key-free интерфейс (ключей наружу не отдаём) ---
def text_enabled():
    return bool(LLM_API_KEYS)


def embed_enabled():
    return bool(EMBED_API_KEYS)


def today():
    return _today()


def key_counts():
    """Сколько ключей в пулах (для /status — без раскрытия самих ключей)."""
    return {"text": len(LLM_API_KEYS), "emb": len(EMBED_API_KEYS)}


async def _has_budget(models, keys, kind):
    today = _today()
    for m, budget in models:
        for idx in range(len(keys)):
            if (await get_usage(f"{today}:{kind}:{m}:k{idx}")) < budget:
                return True
    return False


async def text_budget_left(purpose="autofill"):
    """Есть ли ещё суточный бюджет на текстовые запросы данного profile."""
    return bool(LLM_API_KEYS) and await _has_budget(TEXT_PROFILES[purpose], LLM_API_KEYS, "text")


async def embed_budget_left():
    return bool(EMBED_API_KEYS) and await _has_budget(EMBED_MODELS, EMBED_API_KEYS, "emb")


async def quota_snapshot():
    """[{kind, model, key, used, budget}] — расход по парам ключ×модель за сегодня (для /quota)."""
    today = _today()
    rows = []
    for kind, models, keys in (("text", TEXT_PROFILES["autofill"], LLM_API_KEYS),
                               ("emb", EMBED_MODELS, EMBED_API_KEYS)):
        for m, budget in models:
            for idx in range(len(keys)):
                rows.append({"kind": kind, "model": m, "key": idx, "budget": budget,
                             "used": await get_usage(f"{today}:{kind}:{m}:k{idx}")})
    return rows
