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

# --- Живая лента активности (каждый запрос текста/эмбеддинга к Gemini) ---
FEED_ON = os.getenv("TELEGRAM_FEED", "false").lower() == "true"
# Куда слать ленту (по умолчанию — туда же, куда алерты). Обычно личка админа.
FEED_CHAT_ID = os.getenv("TELEGRAM_FEED_CHAT_ID", "") or TELEGRAM_CHAT_ID
_feed_q = None    # asyncio.Queue — события ленты (создаётся воркером)

_last_sent = {}   # dedup_key -> monotonic-метка последней отправки
_tasks = set()    # держим ссылки на фоновые задачи отправки (иначе их соберёт GC)


def enabled():
    return TELEGRAM_ENABLED and bool(TELEGRAM_BOT_TOKEN) and bool(TELEGRAM_CHAT_ID)


def feed(text):
    """Положить событие в ленту активности (fire-and-forget). Воркер коалесит и шлёт
    пачками раз в несколько секунд, чтобы не упереться в лимиты Telegram. Не бросает."""
    if not FEED_ON or not TELEGRAM_BOT_TOKEN or _feed_q is None or not FEED_CHAT_ID:
        return
    try:
        _feed_q.put_nowait(text)
    except Exception:
        pass  # очередь переполнена — событие просто теряем


async def feed_worker():
    """Шлёт ленту: коалесит до 20 событий в одно сообщение, раз в ~3с (≤20 сообщений/мин —
    безопасно для лимитов Telegram). Стартует из main.py."""
    global _feed_q
    _feed_q = asyncio.Queue(maxsize=2000)
    while True:
        text = await _feed_q.get()
        batch = [text]
        while not _feed_q.empty() and len(batch) < 20:
            try:
                batch.append(_feed_q.get_nowait())
            except Exception:
                break
        await api("sendMessage", {"chat_id": FEED_CHAT_ID, "text": "\n".join(batch),
                                  "disable_web_page_preview": True})
        await asyncio.sleep(3)


async def api(method, payload):
    """Низкоуровневый вызов Telegram Bot API. Возвращает dict ответа или None."""
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
