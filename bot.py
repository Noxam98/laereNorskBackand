"""Интерактивный Telegram-бот администрирования (long-polling).

Принимает команды ТОЛЬКО из админ-чатов (TELEGRAM_ADMIN_IDS), остальные игнорирует.
Запускается фоновой задачей в main.py. Алерты об ошибках шлёт notify.py (в группу),
а бот отвечает в тот чат, откуда пришла команда (личка админа)."""
import os
import asyncio
from config import logger
import notify
from db import (
    get_pool_stats, count_users, get_usage_like, clear_query_cache,
    delete_pool_word, normalize_word, get_usage,
)
import autofill
import llm

ADMIN_IDS = {int(x) for x in (os.getenv("TELEGRAM_ADMIN_IDS", "") or "").replace(" ", "").split(",") if x.strip().lstrip("-").isdigit()}

_tasks = set()  # держим ссылки на фоновые задачи действий


def enabled():
    return bool(notify.TELEGRAM_BOT_TOKEN) and bool(ADMIN_IDS)


def _track(coro):
    t = asyncio.create_task(coro)
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)
    return t


async def _send(chat_id, text, reply_markup=None):
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    await notify.api("sendMessage", payload)


def _menu_markup():
    return {"inline_keyboard": [
        [{"text": "📊 Статус", "callback_data": "stats"}, {"text": "📈 Квота", "callback_data": "quota"}],
        [{"text": "📚 Пул", "callback_data": "pool"}, {"text": "👤 Юзеры", "callback_data": "users"}],
        [{"text": "▶️ Autofill on", "callback_data": "autofill_on"}, {"text": "⏹ Autofill off", "callback_data": "autofill_off"}],
        [{"text": "⏸ Пауза", "callback_data": "pause"}, {"text": "⏯ Возобновить", "callback_data": "resume"}],
        [{"text": "⚡ +10 слов", "callback_data": "gen10"}, {"text": "📝 Описания", "callback_data": "describe"}],
        [{"text": "🌍 Переводы", "callback_data": "translate"}, {"text": "🧹 Кэш", "callback_data": "clearcache"}],
        [{"text": "🔁 Reembed → v2", "callback_data": "reembed2"}],
        [{"text": "🔕 Mute 1ч", "callback_data": "mute60"}, {"text": "🔔 Unmute", "callback_data": "unmute"}],
    ]}


# --- Обработчики команд: возвращают (text, reply_markup|None) ---
async def cmd_help(args, chat_id):
    return (
        "🛠 Админ-бот LearnNorsk\n\n"
        "📊 /stats — расход Gemini, пул, юзеры\n"
        "📈 /quota — остаток дневного бюджета по ключам×моделям\n"
        "📚 /pool — размер пула и что недозаполнено\n"
        "👤 /users — число пользователей\n"
        "ℹ️ /status — состояние фоновых очередей и алертов\n\n"
        "🎛 /autofill on|off — генерация новых слов\n"
        "⏸ /pause · ⏯ /resume — все фоновые очереди\n\n"
        "⚡ /generate N — сгенерировать N слов сейчас\n"
        "📝 /describe — добить описания\n"
        "🌍 /translate — добить переводы\n"
        "🧹 /clearcache — очистить кэш генерации\n"
        "🗑 /delete <слово> — удалить слово из пула\n"
        "🔁 /reembed2 — пере-эмбеддить весь пул на gemini-embedding-2\n\n"
        "🔕 /mute [мин] · 🔔 /unmute — заглушить алерты\n"
        "⏱ /cooldown <сек> — частота повторных алертов\n"
        "📋 /menu — кнопки",
        _menu_markup(),
    )


async def cmd_menu(args, chat_id):
    return "📋 Меню администрирования:", _menu_markup()


def _fmt_usage(rows):
    if not rows:
        return "  (нет расхода)"
    return "\n".join(f"  {k.split(':', 1)[1]}: {v}" for k, v in sorted(rows.items()))


async def cmd_stats(args, chat_id):
    today = llm._today()
    usage = await get_usage_like(today)
    ps = await get_pool_stats()
    users = await count_users()
    txt = {k: v for k, v in usage.items() if ":text:" in k}
    emb = {k: v for k, v in usage.items() if ":emb:" in k}
    return (
        f"📊 Сегодня ({today} UTC)\n\n"
        f"🔤 Текст (LLM):\n{_fmt_usage(txt)}\n\n"
        f"🧮 Эмбеддинги:\n{_fmt_usage(emb)}\n\n"
        f"📚 Пул: {ps['total']} слов · эмб {ps['embedding']} · озв {ps['tts']} · "
        f"опис {ps['description']} · класс {ps['classified']}\n"
        f"👤 Пользователей: {users}", None)


async def cmd_quota(args, chat_id):
    today = llm._today()
    lines = ["📈 Остаток дневного бюджета (использовано/лимит на ключ×модель):", "", "🔤 Текст:"]
    for m, budget in autofill.TEXT_MODELS:
        for i, _ in enumerate(llm.LLM_API_KEYS):
            used = await get_usage(f"{today}:text:{m}:k{i}")
            lines.append(f"  {m} k{i}: {used}/{budget}")
    lines += ["", "🧮 Эмбеддинги:"]
    for m, budget in autofill.EMBED_MODELS:
        for i, _ in enumerate(llm.EMBED_API_KEYS):
            used = await get_usage(f"{today}:emb:{m}:k{i}")
            lines.append(f"  {m} k{i}: {used}/{budget}")
    return "\n".join(lines), None


async def cmd_pool(args, chat_id):
    ps = await get_pool_stats()
    miss_emb = len(await autofill.pool_missing_embedding(10000))
    miss_tts = len(await autofill.pool_missing_tts(10000))
    miss_desc = len(await autofill.pool_missing_description(10000))
    miss_meta = len(await autofill.pool_missing_meta(10000))
    sem_pending = len(await autofill.sem_embed_pending(10000))
    return (
        f"📚 Пул: {ps['total']} слов\n\n"
        f"Недозаполнено:\n"
        f"  🧮 без эмбеддинга: {miss_emb}\n"
        f"  🔊 без озвучки: {miss_tts}\n"
        f"  📝 без описания: {miss_desc}\n"
        f"  🏷 без уровня/тем: {miss_meta}\n"
        f"  🔁 ждут пере-эмбеддинга: {sem_pending}", None)


async def cmd_users(args, chat_id):
    return f"👤 Пользователей: {await count_users()}", None


async def cmd_status(args, chat_id):
    r = autofill.RUNTIME
    mute = notify.mute_left()
    return (
        "ℹ️ Состояние\n\n"
        f"  ⚙️ autofill (генерация): {'ON' if r['autofill'] else 'OFF'}\n"
        f"  ⏸ глобальная пауза: {'ДА' if r['paused'] else 'нет'}\n"
        f"  🔔 алерты: {'🔕 mute ' + str(mute) + 'с' if mute else 'включены'}\n"
        f"  ⏱ cooldown алертов: {notify.NOTIFY_COOLDOWN_SEC}с\n"
        f"  🔑 ключей: LLM {len(llm.LLM_API_KEYS)}, emb {len(llm.EMBED_API_KEYS)}", None)


async def cmd_autofill(args, chat_id):
    val = (args[0].lower() if args else "")
    if val not in ("on", "off"):
        return "Использование: /autofill on|off", None
    autofill.RUNTIME["autofill"] = (val == "on")
    return f"⚙️ autofill: {'ON' if val == 'on' else 'OFF'}", None


async def cmd_pause(args, chat_id):
    autofill.RUNTIME["paused"] = True
    return "⏸ Все фоновые очереди на паузе.", None


async def cmd_resume(args, chat_id):
    autofill.RUNTIME["paused"] = False
    return "⏯ Фоновые очереди возобновлены.", None


async def cmd_generate(args, chat_id):
    try:
        n = max(1, min(50, int(args[0]))) if args else 10
    except ValueError:
        return "Использование: /generate N (1..50)", None
    added, err = await autofill.generate_now(n)
    if err:
        return f"⚠️ Генерация не удалась: {err}", None
    return f"⚡ Добавлено новых слов: {added}", None


async def _bg_action(chat_id, coro, label):
    try:
        res = await coro
        await _send(chat_id, f"✅ {label}: готово ({res})")
    except Exception as e:
        await _send(chat_id, f"⚠️ {label}: ошибка — {str(e)[:200]}")


async def cmd_describe(args, chat_id):
    _track(_bg_action(chat_id, autofill.describe_all_task(), "Описания"))
    return "📝 Добивка описаний запущена — пришлю итог.", None


async def cmd_translate(args, chat_id):
    _track(_bg_action(chat_id, autofill.translate_all_task(), "Переводы"))
    return "🌍 Добивка переводов запущена — пришлю итог.", None


_reembed_running = False


async def cmd_reembed2(args, chat_id):
    global _reembed_running
    if _reembed_running:
        return "⏳ Пере-эмбеддинг уже идёт — дождись отчёта о завершении.", None

    async def report(t):
        await _send(chat_id, t)

    async def runner():
        global _reembed_running
        _reembed_running = True
        try:
            await autofill.reembed_all_task(model="gemini-embedding-2", batch=90, sleep_sec=5, report=report)
        except Exception as e:
            await _send(chat_id, f"⚠️ Пере-эмбеддинг упал: {str(e)[:200]}")
        finally:
            _reembed_running = False

    _track(runner())
    return "🔁 Запускаю пере-эмбеддинг всего пула на gemini-embedding-2 (по 90 за запрос, раз в 5с). Отчёт — на каждый батч.", None


async def cmd_clearcache(args, chat_id):
    await clear_query_cache()
    return "🧹 Кэш генерации очищен.", None


async def cmd_delete(args, chat_id):
    if not args:
        return "Использование: /delete <слово>", None
    w = normalize_word(" ".join(args))
    await delete_pool_word(w)
    return f"🗑 Удалено из пула (и кэша): {w}", None


async def cmd_mute(args, chat_id):
    minutes = 60
    if args:
        try:
            minutes = max(1, int(args[0]))
        except ValueError:
            return "Использование: /mute [минуты]", None
    notify.mute(minutes * 60)
    return f"🔕 Алерты заглушены на {minutes} мин.", None


async def cmd_unmute(args, chat_id):
    notify.unmute()
    return "🔔 Алерты снова включены.", None


async def cmd_cooldown(args, chat_id):
    if not args:
        return f"Текущий cooldown: {notify.NOTIFY_COOLDOWN_SEC}с. Использование: /cooldown <сек>", None
    try:
        sec = max(0, int(args[0]))
    except ValueError:
        return "Использование: /cooldown <секунды>", None
    notify.set_cooldown(sec)
    return f"⏱ Cooldown алертов: {sec}с.", None


COMMANDS = {
    "start": cmd_help, "help": cmd_help, "menu": cmd_menu,
    "stats": cmd_stats, "quota": cmd_quota, "pool": cmd_pool, "users": cmd_users, "status": cmd_status,
    "autofill": cmd_autofill, "pause": cmd_pause, "resume": cmd_resume,
    "generate": cmd_generate, "describe": cmd_describe, "translate": cmd_translate,
    "clearcache": cmd_clearcache, "delete": cmd_delete, "reembed2": cmd_reembed2,
    "mute": cmd_mute, "unmute": cmd_unmute, "cooldown": cmd_cooldown,
}

# Кнопки меню → текстовая команда
BUTTONS = {
    "stats": "/stats", "quota": "/quota", "pool": "/pool", "users": "/users", "status": "/status",
    "autofill_on": "/autofill on", "autofill_off": "/autofill off",
    "pause": "/pause", "resume": "/resume",
    "gen10": "/generate 10", "describe": "/describe", "translate": "/translate",
    "clearcache": "/clearcache", "reembed2": "/reembed2", "mute60": "/mute 60", "unmute": "/unmute",
}


async def _dispatch(text, chat_id):
    cmd = text.strip().split()[0].lstrip("/").split("@")[0].lower()
    args = text.strip().split()[1:]
    handler = COMMANDS.get(cmd)
    if not handler:
        return f"Неизвестная команда: /{cmd}. /menu — список.", None
    return await handler(args, chat_id)


async def _process(update):
    msg = update.get("message")
    cb = update.get("callback_query")
    if cb:
        chat_id = (cb.get("message") or {}).get("chat", {}).get("id")
        from_id = (cb.get("from") or {}).get("id")
        if from_id not in ADMIN_IDS:
            await notify.api("answerCallbackQuery", {"callback_query_id": cb["id"], "text": "Нет доступа"})
            return
        await notify.api("answerCallbackQuery", {"callback_query_id": cb["id"]})
        text = BUTTONS.get(cb.get("data") or "", "")
        if text:
            reply, markup = await _dispatch(text, chat_id)
            await _send(chat_id, reply, markup)
        return
    if msg:
        chat_id = msg.get("chat", {}).get("id")
        from_id = (msg.get("from") or {}).get("id")
        text = msg.get("text") or ""
        if from_id not in ADMIN_IDS:
            return  # тихо игнорируем чужих
        if not text.startswith("/"):
            return
        reply, markup = await _dispatch(text, chat_id)
        await _send(chat_id, reply, markup)


async def poll_loop():
    """Long-polling Telegram getUpdates. Один потребитель на инстанс."""
    if not enabled():
        logger.info("telegram bot: OFF (нет TELEGRAM_ADMIN_IDS)")
        return
    logger.info(f"telegram bot: ON, админов {len(ADMIN_IDS)}")
    offset = None
    while True:
        try:
            payload = {"timeout": 25, "allowed_updates": ["message", "callback_query"]}
            if offset is not None:
                payload["offset"] = offset
            data = await notify.api("getUpdates", payload)
            for upd in (data or {}).get("result", []):
                offset = upd["update_id"] + 1
                try:
                    await _process(upd)
                except Exception as e:
                    logger.warning(f"bot process update: {e}")
        except Exception as e:
            logger.warning(f"bot poll_loop: {e}")
            await asyncio.sleep(5)
