"""Конфиг провайдера: эндпоинты, пулы ключей, профили моделей с бюджетами.
Только данные из окружения — ни запросов, ни логики (это нижний слой пакета llm)."""
import os

# --- Несекретная конфигурация моделей/эндпоинтов ---
# Это НЕ секреты, поэтому реальные значения живут здесь, под версионированием (а не в Fly-секретах).
# env-переменные оставлены опциональным override (другой провайдер/локальные эксперименты),
# но дефолты — рабочие прод-значения, так что без env всё работает само.
GEMINI_OPENAI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", GEMINI_OPENAI_URL)
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")   # профиль autofill (массовый фон)
USER_TEXT_MODEL = "gemini-3.5-flash"                          # профиль user (интерактив: описания, генерация слов)
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", GEMINI_OPENAI_URL)
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-2")


def _parse_keys(multi_env, single_env):
    """Пул API-ключей (несколько Google-аккаунтов = суммарно больше квоты).
    Ключи перечисляются через запятую — работает и LLM_API_KEYS, и одиночный LLM_API_KEY."""
    raw = (os.getenv(multi_env, "") or "").strip() or (os.getenv(single_env, "") or "").strip()
    return [k.strip() for k in raw.split(",") if k.strip()]


LLM_API_KEYS = _parse_keys("LLM_API_KEYS", "LLM_API_KEY")
EMBED_API_KEYS = _parse_keys("EMBED_API_KEYS", "EMBED_API_KEY")
# Первый ключ пула — для инициализации клиента (далее ключ подставляется на каждый запрос).
LLM_API_KEY = LLM_API_KEYS[0] if LLM_API_KEYS else ""
EMBED_API_KEY = EMBED_API_KEYS[0] if EMBED_API_KEYS else ""


def _parse_models(env, default_model):
    """Список моделей из env "model[,model...]" (приоритет — слева направо). Старый формат
    "model:rpd" поддерживается для совместимости — число-бюджет после ':' просто отбрасываем."""
    out = []
    for part in (os.getenv(env, "") or "").split(","):
        part = part.strip()
        if not part:
            continue
        head, sep, tail = part.rpartition(":")
        out.append(head.strip() if (sep and tail.strip().isdigit()) else part)
    return out or [default_model]


# purpose → список моделей в порядке приоритета (запасные, если основная отвалится по 429).
TEXT_PROFILES = {
    "user": _parse_models("USER_TEXT_MODELS", USER_TEXT_MODEL),
    "autofill": _parse_models("AUTOFILL_TEXT_MODELS", LLM_MODEL),
}
EMBED_MODELS = _parse_models("AUTOFILL_EMBED_MODELS", EMBED_MODEL)
