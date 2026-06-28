"""Веб-пуши: подписки + напоминание «13 часов бездействия».

Полностью изолировано и FAIL-SAFE: если VAPID-ключи не заданы в окружении — воркер пустой
(no-op), эндпоинты отдают «не сконфигурировано». Импорт pywebpush ленивый: нет пакета —
отправка молча выключается, приложение не падает. Своя таблица push_subscriptions, своё
соединение aiosqlite — ничего из остального бэка не задевает.

Логика напоминания: раз в ~30 мин ищем подписки юзеров, у кого последняя активность
(MAX(user_words.last_seen)) старше IDLE_HOURS, и кому ещё НЕ слали с момента той активности
(last_reminded_at < last_active) → один пуш и отметка last_reminded_at. Перезанялся → перевзвод.
Тихие часы (Europe/Oslo 22:00–08:00) пуши не шлём.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from config import logger
from db import DATABASE_URL
from auth import get_current_user

VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "mailto:maksym.melnykov@sourceit.no")
IDLE_HOURS = int(os.getenv("PUSH_IDLE_HOURS", "13"))
CHECK_INTERVAL_SEC = int(os.getenv("PUSH_CHECK_INTERVAL_SEC", "1800"))  # как часто опрашивать (30 мин)
QUIET_FROM = int(os.getenv("PUSH_QUIET_FROM", "22"))   # тихие часы (Oslo): не слать с 22:00…
QUIET_TO = int(os.getenv("PUSH_QUIET_TO", "8"))        # …до 08:00

REMINDER = {
    "title": "Пора и меру знать 😄",
    "body": "13 часов без норвежского — это уже мини-отпуск. Заглянешь на пару слов?",
    "url": "/#/learning",
}

MOD_THROTTLE_SEC = int(os.getenv("PUSH_MOD_THROTTLE_SEC", "900"))  # не чаще раза в 15 мин
_LAST_MOD_NOTIFY = {"at": None}


def configured():
    return bool(VAPID_PRIVATE_KEY and VAPID_PUBLIC_KEY)


_VAPID_OBJ = None


def _vapid_obj():
    """Загруженный Vapid-ключ (кэш). pywebpush НЕ умеет грузить PEM из строки сам
    (Could not deserialize key data) — передаём ему готовый объект."""
    global _VAPID_OBJ
    if _VAPID_OBJ is None and VAPID_PRIVATE_KEY:
        from py_vapid import Vapid01
        if "-----" in VAPID_PRIVATE_KEY:
            _VAPID_OBJ = Vapid01.from_pem(VAPID_PRIVATE_KEY.encode())
        else:
            _VAPID_OBJ = Vapid01.from_raw(VAPID_PRIVATE_KEY.encode())
    return _VAPID_OBJ


# ---------- хранилище подписок (своя таблица, своё соединение) ----------
async def _exec(sql, params=()):
    async with aiosqlite.connect(DATABASE_URL) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()
        await db.commit()
        return rows


# Доверенные хосты push-провайдеров. endpoint приходит от клиента, и бэкенд САМ шлёт на него POST
# (фоновый воркер) — без проверки это SSRF (можно подсунуть internal-адрес). Принимаем только https
# на известные сервисы пушей; всё прочее отвергаем (fail-closed; новый провайдер — добавить суффикс).
_PUSH_HOST_SUFFIXES = (
    ".googleapis.com",             # Chrome/Edge (FCM: fcm.googleapis.com)
    ".push.services.mozilla.com",  # Firefox
    ".notify.windows.com",         # Edge/Windows (WNS)
    ".push.apple.com",             # Safari/Apple
)


def _valid_push_endpoint(endpoint: str) -> bool:
    from urllib.parse import urlparse
    try:
        u = urlparse(endpoint or "")
    except Exception:
        return False
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host.endswith(s) for s in _PUSH_HOST_SUFFIXES)


async def save_subscription(user_id, sub):
    info = sub.get("keys") or {}
    endpoint = sub.get("endpoint")
    p256dh = info.get("p256dh")
    auth = info.get("auth")
    if not (endpoint and p256dh and auth):
        raise ValueError("bad subscription")
    if not _valid_push_endpoint(endpoint):   # SSRF-защита: только https на известные push-сервисы
        raise ValueError("bad endpoint")
    now = datetime.utcnow().isoformat()
    # один и тот же endpoint может перепривязываться к юзеру — обновляем ключи/владельца
    await _exec(
        """INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, last_reminded_at, created_at)
           VALUES (?, ?, ?, ?, NULL, ?)
           ON CONFLICT(endpoint) DO UPDATE SET user_id=excluded.user_id, p256dh=excluded.p256dh,
               auth=excluded.auth""",
        (user_id, endpoint, p256dh, auth, now),
    )


async def delete_subscription(endpoint, user_id):
    # owner-scoped: юзер удаляет ТОЛЬКО свою подписку (иначе по чужому endpoint можно отписать другого)
    await _exec("DELETE FROM push_subscriptions WHERE endpoint = ? AND user_id = ?", (endpoint, user_id))


# ---------- отправка (ленивый pywebpush) ----------
def _send_sync(endpoint, p256dh, auth, payload):
    """Блокирующая отправка одного пуша. Возвращает (ok, gone). gone=True → подписку удалить."""
    try:
        from pywebpush import webpush, WebPushException
    except Exception as e:  # пакет не установлен — отправка выключена, но без падения
        logger.warning(f"pywebpush недоступен: {e}")
        return (False, False)
    try:
        webpush(
            subscription_info={"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}},
            data=json.dumps(payload),
            vapid_private_key=_vapid_obj(),
            vapid_claims={"sub": VAPID_SUBJECT},
            timeout=20,
        )
        return (True, False)
    except WebPushException as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        gone = code in (404, 410)  # подписка протухла — чистим
        if not gone:
            logger.warning(f"webpush error {code}: {str(e)[:160]}")
        return (False, gone)
    except Exception as e:
        logger.warning(f"webpush send error: {str(e)[:160]}")
        return (False, False)


async def _send(endpoint, p256dh, auth, payload):
    return await asyncio.to_thread(_send_sync, endpoint, p256dh, auth, payload)


def _quiet_now():
    """True — сейчас тихие часы (Oslo), пуши не шлём."""
    try:
        from zoneinfo import ZoneInfo
        hour = datetime.now(ZoneInfo("Europe/Oslo")).hour
    except Exception:
        hour = (datetime.utcnow().hour + 1) % 24  # грубый фолбэк ~Oslo
    # окно тишины может пересекать полночь (22..8)
    if QUIET_FROM <= QUIET_TO:
        return QUIET_FROM <= hour < QUIET_TO
    return hour >= QUIET_FROM or hour < QUIET_TO


# ---------- пуш админам: новые слова на модерации ----------
async def notify_moderators(pending_count):
    """Уведомить админов (web-push), что появились слова на модерации. Троттлинг (не чаще раза
    в MOD_THROTTLE_SEC) + тихие часы. Fail-safe: при любой ошибке/не-конфиге молча выходим."""
    try:
        if not configured():
            return
        from auth import ADMIN_USERS
        if not ADMIN_USERS:
            return
        if _quiet_now():
            return
        now = datetime.utcnow()
        last = _LAST_MOD_NOTIFY["at"]
        if last and (now - last).total_seconds() < MOD_THROTTLE_SEC:
            return
        marks = ",".join("?" for _ in ADMIN_USERS)
        rows = await _exec(
            f"SELECT s.endpoint, s.p256dh, s.auth FROM push_subscriptions s "
            f"JOIN users u ON u.id = s.user_id WHERE lower(u.username) IN ({marks})",
            tuple(ADMIN_USERS),
        )
        if not rows:
            return
        payload = {
            "title": "Слова на модерации",
            "body": f"Новых слов в очереди: {pending_count}",
            "url": "/#/moderation",
        }
        sent = False
        for r in rows:
            ok, gone = await _send(r["endpoint"], r["p256dh"], r["auth"], payload)
            if gone:
                await delete_subscription(r["endpoint"])
            elif ok:
                sent = True
        if sent:
            _LAST_MOD_NOTIFY["at"] = now
    except Exception as e:
        logger.warning(f"notify_moderators: {str(e)[:160]}")


# ---------- воркер напоминаний ----------
async def reminder_loop():
    await asyncio.sleep(60)  # дать приложению подняться
    if not configured():
        logger.info("webpush: VAPID не задан — напоминания выключены")
    while True:
        try:
            if not configured():
                await asyncio.sleep(3600)  # не сконфигурировано — спим, ничего не делаем
                continue
            if _quiet_now():
                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue
            cutoff = (datetime.utcnow() - timedelta(hours=IDLE_HOURS)).isoformat()
            rows = await _exec(
                """SELECT s.id AS sid, s.endpoint, s.p256dh, s.auth, s.last_reminded_at,
                          MAX(uw.last_seen) AS last_active
                   FROM push_subscriptions s
                   JOIN user_words uw ON uw.user_id = s.user_id
                   GROUP BY s.id
                   HAVING last_active IS NOT NULL
                      AND last_active <= ?
                      AND (s.last_reminded_at IS NULL OR s.last_reminded_at < last_active)""",
                (cutoff,),
            )
            now = datetime.utcnow().isoformat()
            for r in rows:
                ok, gone = await _send(r["endpoint"], r["p256dh"], r["auth"], REMINDER)
                if gone:
                    await delete_subscription(r["endpoint"])
                elif ok:
                    await _exec("UPDATE push_subscriptions SET last_reminded_at = ? WHERE id = ?", (now, r["sid"]))
            if rows:
                logger.info(f"webpush: разослано напоминаний — {len(rows)}")
        except Exception as e:
            logger.warning(f"reminder_loop: {str(e)[:200]}")
        await asyncio.sleep(CHECK_INTERVAL_SEC)


# ---------- API ----------
router = APIRouter(prefix="/push", tags=["push"])


@router.get("/vapid")
async def vapid_public():
    if not VAPID_PUBLIC_KEY:
        raise HTTPException(status_code=503, detail="push not configured")
    return {"publicKey": VAPID_PUBLIC_KEY}


@router.post("/subscribe")
async def subscribe(body: dict, user=Depends(get_current_user)):
    try:
        await save_subscription(user["id"], body)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad subscription")
    return {"ok": True}


@router.post("/unsubscribe")
async def unsubscribe(body: dict, user=Depends(get_current_user)):
    ep = body.get("endpoint")
    if ep:
        await delete_subscription(ep, user["id"])
    return {"ok": True}
