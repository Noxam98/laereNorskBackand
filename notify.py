import os
import time
import asyncio
from config import logger

# --- Уведомления в Telegram (бот). Конфиг через env; см. .env. ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
# Один и тот же тип события не чаще, чем раз в N секунд — чтобы сбой не спамил сотнями сообщений.
NOTIFY_COOLDOWN_SEC = int(os.getenv("NOTIFY_COOLDOWN_SEC", "900"))

_last_sent = {}   # dedup_key -> monotonic-метка последней отправки
_tasks = set()    # держим ссылки на фоновые задачи отправки (иначе их соберёт GC)


def enabled():
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


def _throttled(key):
    """True — если по этому ключу недавно уже слали (внутри кулдауна)."""
    now = time.monotonic()
    last = _last_sent.get(key)
    if last is not None and (now - last) < NOTIFY_COOLDOWN_SEC:
        return True
    _last_sent[key] = now
    return False


async def _send(text):
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "disable_web_page_preview": True,
            })
            if r.status_code != 200:
                logger.warning(f"telegram notify failed: {r.status_code} {r.text[:200]}")
    except Exception as e:
        # Уведомление не должно ронять основную логику — глотаем любые ошибки отправки.
        logger.warning(f"telegram notify error: {e}")


def notify(text, dedup_key=None):
    """Fire-and-forget уведомление в Telegram. Не блокирует вызывающий код и не бросает.
    dedup_key — чтобы повторяющийся сбой не спамил (кулдаун NOTIFY_COOLDOWN_SEC)."""
    if not enabled():
        return
    if _throttled(dedup_key or text):
        return
    try:
        task = asyncio.create_task(_send(text))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    except RuntimeError:
        # нет активного event loop (например, вызов вне приложения) — просто пропускаем
        pass
