"""Лёгкий in-memory rate-limit (приложение на ОДНОЙ машине Fly — состояние в памяти допустимо).
Скользящее окно по ключу. Защищает дорогие LLM/TTS-эндпоинты от цикл-абьюза: исчерпания общей
квоты Gemini (DoS для всех пользователей) и роста стоимости провайдера. Лимиты щедрые — нормальное
использование их не задевает; настраиваются через окружение.
"""
import os
import time
from collections import defaultdict, deque
from fastapi import Depends, HTTPException, Request

from auth import get_current_user

_hits = defaultdict(deque)   # ключ -> deque[timestamps]


def _hit(key, max_n, window):
    now = time.monotonic()
    dq = _hits[key]
    cut = now - window
    while dq and dq[0] < cut:
        dq.popleft()
    if len(dq) >= max_n:
        retry = max(1, int(dq[0] + window - now))
        raise HTTPException(status_code=429, detail="Слишком много запросов, попробуйте чуть позже",
                            headers={"Retry-After": str(retry)})
    dq.append(now)


def client_ip(request: Request) -> str:
    # Fly-Client-IP выставляет доверенный edge Fly; X-Forwarded-For клиент может подделать (обход
    # лимита по IP спуфингом заголовка) — потому доверенный заголовок в приоритете.
    fly = request.headers.get("fly-client-ip")
    if fly:
        return fly.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Дорогие LLM-вызовы (генерация/описание/вопрос/правка/диф/OCR/импорт) — лимит НА ПОЛЬЗОВАТЕЛЯ.
_LLM_MAX = int(os.getenv("RATE_LLM_MAX", "40"))
_LLM_WINDOW = int(os.getenv("RATE_LLM_WINDOW", "600"))   # дефолт: 40 запросов / 10 минут

# Глобальный бэкстоп на общую квоту Gemini: пер-юзер лимит не спасает от НЕСКОЛЬКИХ аккаунтов,
# суммарно множащих нагрузку. ЩЕДРЫЙ потолок на ВСЕХ юзеров сразу — нормальная многопользовательская
# нагрузка его не задевает, но убегающий цикл/рой аккаунтов упрётся. Оба env-переопределяемы.
_LLM_GLOBAL_MAX = int(os.getenv("RATE_LLM_GLOBAL_MAX", "600"))
_LLM_GLOBAL_WINDOW = int(os.getenv("RATE_LLM_GLOBAL_WINDOW", "600"))   # дефолт: 600 запросов / 10 минут на ВСЕХ


async def llm_rate_limit(user=Depends(get_current_user)):
    """Замена get_current_user для дорогих LLM-эндпоинтов: та же аутентификация + лимит на юзера
    и щедрый глобальный бэкстоп на общую квоту Gemini."""
    _hit(("llm", user["id"]), _LLM_MAX, _LLM_WINDOW)
    _hit(("llm_global",), _LLM_GLOBAL_MAX, _LLM_GLOBAL_WINDOW)
    return user


# TTS — публичный эндпоинт, лимит ПО IP (щедрый: аудио кешируется клиентом, реальных синтезов мало).
_TTS_MAX = int(os.getenv("RATE_TTS_MAX", "1200"))
_TTS_WINDOW = int(os.getenv("RATE_TTS_WINDOW", "300"))   # дефолт: 1200 / 5 минут на IP


def tts_rate_limit(request: Request):
    _hit(("tts", client_ip(request)), _TTS_MAX, _TTS_WINDOW)
