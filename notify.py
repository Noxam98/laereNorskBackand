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
_mute_until = 0.0  # monotonic-метка, до которой алерты заглушены (управляется ботом)


def enabled():
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


# --- Рантайм-управление алертами (из Telegram-бота) ---
def mute(seconds):
    global _mute_until
    _mute_until = time.monotonic() + max(0, seconds)


def unmute():
    global _mute_until
    _mute_until = 0.0


def is_muted():
    return time.monotonic() < _mute_until


def mute_left():
    """Сколько секунд ещё заглушено (0 — не заглушено)."""
    return max(0, int(_mute_until - time.monotonic()))


def set_cooldown(seconds):
    global NOTIFY_COOLDOWN_SEC
    NOTIFY_COOLDOWN_SEC = max(0, int(seconds))


async def api(method, payload):
    """Низкоуровневый вызов Telegram Bot API. Возвращает dict ответа или None.
    Используют и алерты (_send), и интерактивный бот (bot.py)."""
    if not TELEGRAM_BOT_TOKEN:
        return None
    try:
        import httpx
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
        async with httpx.AsyncClient(timeout=40) as client:
            r = await client.post(url, json=payload)
            data = r.json()
            if not data.get("ok"):
                logger.warning(f"telegram {method} failed: {r.status_code} {str(data)[:200]}")
            return data
    except Exception as e:
        logger.warning(f"telegram {method} error: {e}")
        return None


def _throttled(key):
    """True — если по этому ключу недавно уже слали (внутри кулдауна)."""
    now = time.monotonic()
    last = _last_sent.get(key)
    if last is not None and (now - last) < NOTIFY_COOLDOWN_SEC:
        return True
    _last_sent[key] = now
    return False


async def _send(text):
    await api("sendMessage", {"chat_id": TELEGRAM_CHAT_ID, "text": text,
                              "disable_web_page_preview": True})


def notify(text, dedup_key=None):
    """Fire-and-forget уведомление в Telegram. Не блокирует вызывающий код и не бросает.
    dedup_key — чтобы повторяющийся сбой не спамил (кулдаун NOTIFY_COOLDOWN_SEC)."""
    if not enabled() or is_muted():
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
