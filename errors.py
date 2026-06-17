import re
import json
from dataclasses import dataclass
from config import logger
import notify

# --- Классификация ошибок LLM/эмбеддинг-провайдера (Gemini через OpenAI-совместимый клиент) ---
# Раньше тип ошибки угадывался по подстроке в тексте ("429"/"quota"). Теперь — по типам
# исключений openai SDK + статус-коду, с единой трактовкой: какой HTTP отдать клиенту,
# что написать в лог и слать ли уведомление в Telegram.

QUOTA = "quota"            # 429 / RESOURCE_EXHAUSTED — кончилась дневная квота или RPM
AUTH = "auth"              # 401/403 — неверный/протухший ключ, нет доступа
TIMEOUT = "timeout"        # запрос не уложился в таймаут
CONNECTION = "connection"  # сеть/DNS — провайдер недоступен
SERVER = "server"          # 5xx на стороне провайдера
BAD_REQUEST = "bad_request"  # 400 — наш запрос отклонён (например, schema не поддержана)
UNKNOWN = "unknown"        # провайдерская ошибка неизвестного вида
INTERNAL = "internal"      # НЕ провайдерская: сбой бэкенда (БД, логика и т.п.)


@dataclass
class LlmError:
    kind: str            # один из констант выше
    http_status: int     # что отдать клиенту
    user_detail: str     # текст для клиента (без утечки внутренностей)
    alert: bool          # слать ли уведомление в Telegram
    summary: str         # короткое человекочитаемое описание для лога/Telegram


def _status_of(exc):
    return (getattr(exc, "status_code", None)
            or getattr(getattr(exc, "response", None), "status_code", None))


def classify(exc) -> LlmError:
    """Разобрать исключение от LLM/эмбеддинг-вызова в структурированную ошибку."""
    msg = str(exc)
    low = msg.lower()
    status = _status_of(exc)

    # Типы openai подгружаем лениво — не тащим зависимость в импорт модуля.
    try:
        from openai import (
            APIError, APITimeoutError, APIConnectionError, RateLimitError,
            AuthenticationError, PermissionDeniedError, BadRequestError,
        )
    except Exception:  # на случай иной версии SDK — упадём в строковые эвристики ниже
        APIError = ()
        APITimeoutError = APIConnectionError = RateLimitError = ()
        AuthenticationError = PermissionDeniedError = BadRequestError = ()

    def is_(exc_types):
        return exc_types and isinstance(exc, exc_types)

    # Это вообще ошибка ПРОВАЙДЕРА (Gemini), а не наш внутренний сбой (БД/логика)?
    # Признак — тип из openai SDK либо наличие HTTP-статуса/ответа.
    provider = is_(APIError) or status is not None

    # 1) Квота / rate limit — маркеры специфичны для провайдера, ловим и без типа
    if is_(RateLimitError) or status == 429 \
            or "resource_exhausted" in low or "rate limit" in low \
            or ("quota" in low and provider):
        return LlmError(QUOTA, 429, "Translation provider quota exhausted", True,
                        "Квота/лимит Gemini исчерпаны (429)")
    if provider:
        # 2) Ключ / доступ
        if is_(AuthenticationError) or is_(PermissionDeniedError) or status in (401, 403):
            return LlmError(AUTH, 502, "Translation provider auth error", True,
                            f"Проблема с ключом/доступом Gemini ({status or '401/403'})")
        # 3) Таймаут
        if is_(APITimeoutError):
            return LlmError(TIMEOUT, 504, "Translation provider timeout", True,
                            "Таймаут запроса к Gemini")
        # 4) Сеть / соединение
        if is_(APIConnectionError):
            return LlmError(CONNECTION, 502, "Translation provider unreachable", True,
                            "Сеть/соединение с Gemini недоступно")
        # 5) 5xx на стороне провайдера
        if status and status >= 500:
            return LlmError(SERVER, 502, "Translation provider error", True,
                            f"Сбой провайдера Gemini ({status})")
        # 6) Плохой запрос (наша ошибка/неподдерживаемая фича) — не алёртим, не инцидент
        if is_(BadRequestError) or status == 400:
            return LlmError(BAD_REQUEST, 500, "Translation provider rejected request", False,
                            "Запрос отклонён провайдером (400)")
        return LlmError(UNKNOWN, 502, "Translation provider error", True,
                        f"Неизвестная ошибка Gemini: {msg[:200]}")
    # Не провайдерская — внутренний сбой бэкенда (БД, логика). Алёртим, но честно называем.
    return LlmError(INTERNAL, 500, "Internal error", True,
                    f"Сбой бэкенда: {msg[:200]}")


def report(exc, context) -> LlmError:
    """Классифицировать ошибку, залогировать и (если важно) уведомить в Telegram.
    context — где случилось (имя функции/цикла) для лога и текста уведомления."""
    info = classify(exc)
    (logger.error if info.alert else logger.warning)(f"{context}: [{info.kind}] {exc}")
    if info.alert:
        notify.notify(f"⚠️ LearnNorsk: {info.summary}\nГде: {context}",
                      dedup_key=f"{info.kind}:{context}")
    return info


def _error_body(exc):
    """Распарсить тело ошибки провайдера в dict (из exc.body или из JSON в тексте)."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", body)
    m = re.search(r'(\{.*\})', str(exc), re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d.get("error", d)
        except Exception:
            pass
    return {}


def quota_info(exc):
    """Из 429-ошибки достать {retry, scope, limit}: сколько ждать до сброса (RetryInfo
    retryDelay или из текста), тип лимита (RPD/RPM/TPM по quotaId) и значение потолка.
    retry/scope/limit = None, если не нашли."""
    err = _error_body(exc)
    msg = err.get("message") or str(exc)
    retry = scope = limit = None
    for d in err.get("details", []) or []:
        t = str(d.get("@type", ""))
        if t.endswith("RetryInfo"):
            mm = re.match(r'([\d.]+)s', str(d.get("retryDelay", "")))
            if mm:
                retry = float(mm.group(1))
        if t.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                qid = v.get("quotaId", "")
                if "PerDay" in qid:
                    scope = "RPD"
                elif "PerMinute" in qid:
                    scope = "RPM"
                elif "Token" in qid:
                    scope = "TPM"
                if v.get("quotaValue"):
                    try:
                        limit = int(v["quotaValue"])
                    except (TypeError, ValueError):
                        pass
    if retry is None:  # фолбэк: "Please retry in 42.09s"
        mm = re.search(r'retry in ([\d.]+)s', msg, re.IGNORECASE)
        if mm:
            retry = float(mm.group(1))
    if scope is None:
        low = msg.lower().replace(" ", "")
        scope = "RPD" if "perday" in low else ("RPM" if "perminute" in low else None)
    return {"retry": retry, "scope": scope, "limit": limit}
