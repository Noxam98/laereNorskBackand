"""Конфиг провайдера: эндпоинты, пулы ключей, профили моделей с бюджетами.
Только данные из окружения — ни запросов, ни логики (это нижний слой пакета llm)."""
import os

# --- LLM-провайдер (OpenAI-совместимый): Gemini / Groq / OpenRouter / любой через env ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-r1:free")
# Эмбеддинги (Gemini по умолчанию, OpenAI-совместимый эндпоинт; провайдер через env).
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-004")


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


def _parse_models(env, default_model, default_budget=10 ** 9):
    """Список (модель, суточный_бюджет) из env "model:rpd,...". Бюджет — на ПАРУ ключ×модель."""
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


# purpose → список моделей: "user" — без потолка (живые запросы), "autofill" — с лимитом
# (фон оставляет запас квоты живым запросам днём).
TEXT_PROFILES = {
    "user": _parse_models("USER_TEXT_MODELS", LLM_MODEL),
    "autofill": _parse_models("AUTOFILL_TEXT_MODELS", LLM_MODEL, int(os.getenv("TEXT_DAILY_BUDGET", "400"))),
}
EMBED_MODELS = _parse_models("AUTOFILL_EMBED_MODELS", EMBED_MODEL, int(os.getenv("EMBED_DAILY_BUDGET", "800")))
