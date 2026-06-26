"""Здоровье ключей («дневник 429»). Бюджетов/лимитов больше нет — полагаемся на реальные
429 от провайдера: ключ, получивший 429, скипаем COOLDOWN_SEC секунд (для пары ключ×модель).
Учёт расхода (incr_*) ведём только для статистики, в выборе ключа он не участвует."""
import os
import time
from datetime import datetime
from db import incr_usage
from .settings import LLM_API_KEYS, EMBED_API_KEYS, TEXT_PROFILES, EMBED_MODELS

# Фолбэк-кулдаун, если в ответе нет retryDelay. Обычно ждём ровно retryDelay из 429.
COOLDOWN_SEC = int(os.getenv("KEY_429_COOLDOWN_SEC", "10"))

_blocked = {}    # (kind, model, idx) -> monotonic-время, до которого ключ скипаем после 429
_rr_cursor = {}  # (kind, model) -> индекс последнего успешно использованного ключа (round-robin)


def text_enabled():
    return bool(LLM_API_KEYS)


def embed_enabled():
    return bool(EMBED_API_KEYS)


def today():
    return datetime.utcnow().strftime("%Y-%m-%d")


def key_counts():
    return {"text": len(LLM_API_KEYS), "emb": len(EMBED_API_KEYS)}


def _key_id(key, pool):
    try:
        return pool.index(key)
    except (ValueError, AttributeError):
        return 0


# --- Дневник 429 (по паре МОДЕЛЬ × КЛЮЧ, не по всему ключу) ---
def mark_429(kind, model, idx, seconds=None):
    """Зафиксировать 429 у пары (модель model, ключ idx) — скипаем её `seconds` секунд
    (из retryDelay ответа), либо COOLDOWN_SEC если время не пришло. Другие модели на этом
    же ключе не затрагиваются."""
    wait = COOLDOWN_SEC if (seconds is None or seconds <= 0) else seconds
    _blocked[(kind, model, idx)] = time.monotonic() + wait


def advance(kind, model, idx):
    """Запомнить успешно использованный ключ — следующий запрос начнём со следующего."""
    _rr_cursor[(kind, model)] = idx


def _fresh(kind, model, idx):
    until = _blocked.get((kind, model, idx))
    return until is None or time.monotonic() >= until


def candidates(models, keys, kind):
    """Порядок попыток [(model, key, idx)]: модели по приоритету, ключи round-robin.
    «Свежие» (без 429 за последние COOLDOWN_SEC) — впереди; «остывающие» — в хвосте по
    времени истечения (чтобы не вставать колом, если все в кулдауне)."""
    n = len(keys)
    fresh, cooling = [], []
    for entry in models:
        m = entry[0] if isinstance(entry, (tuple, list)) else entry
        start = _rr_cursor.get((kind, m), -1)
        for off in range(1, n + 1):
            idx = (start + off) % n
            if _fresh(kind, m, idx):
                fresh.append((m, keys[idx], idx))
            else:
                cooling.append((_blocked[(kind, m, idx)], m, keys[idx], idx))
    cooling.sort(key=lambda x: x[0])  # сначала те, у кого кулдаун истечёт раньше
    return fresh + [(m, k, i) for _, m, k, i in cooling]


def _any_fresh(models, keys, kind):
    n = len(keys)
    for entry in models:
        m = entry[0] if isinstance(entry, (tuple, list)) else entry
        if any(_fresh(kind, m, idx) for idx in range(n)):
            return True
    return False


# --- key-free интерфейс наружу ---
def text_candidates(purpose, model=None):
    # model может быть строкой (одна модель) ИЛИ списком — своя цепочка приоритета (напр. 3.5→lite)
    if model:
        models = list(model) if isinstance(model, (list, tuple)) else [model]
    else:
        models = TEXT_PROFILES[purpose]
    return candidates(models, LLM_API_KEYS, "text")


def embed_candidates():
    return candidates(EMBED_MODELS, EMBED_API_KEYS, "emb")


def text_available(purpose="autofill"):
    """Есть ли хоть один текстовый ключ БЕЗ недавнего 429 (гейт фоновых очередей)."""
    return text_enabled() and _any_fresh(TEXT_PROFILES[purpose], LLM_API_KEYS, "text")


def embed_available():
    return embed_enabled() and _any_fresh(EMBED_MODELS, EMBED_API_KEYS, "emb")


# --- Учёт расхода (только для статистики «Расход Gemini сегодня») ---
async def incr_text(model, key):
    await incr_usage(f"{today()}:text:{model}:k{_key_id(key, LLM_API_KEYS)}")


async def incr_emb(model, key):
    await incr_usage(f"{today()}:emb:{model}:k{_key_id(key, EMBED_API_KEYS)}")
