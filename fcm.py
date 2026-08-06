"""FCM-пуши для Android-приложения (Capacitor) + ДЕДУП с веб-пушами.

Зачем отдельный канал: Web Push внутри Capacitor WebView не работает (нет Service Worker
Push), поэтому напоминание «13ч бездействия» до приложения доезжает только через FCM.
`webpush.py` остаётся каналом ВЕБА как есть — здесь ничего из него не переписывается,
берутся только общие константы (порог простоя, тихие часы, текст напоминания).

Fail-safe как и web-push: нет креденшелов в окружении — канал просто выключен (воркер спит,
эндпоинты подписки продолжают работать), отсутствие google-auth/requests не роняет бэкенд.
Отправка — FCM HTTP v1 (`/v1/projects/{id}/messages:send`) с OAuth2-токеном сервис-аккаунта;
legacy server key Google отключил, его тут нет.

════════════════════ ДЕДУП: один юзер — одно напоминание ════════════════════
ПРИОРИТЕТ КАНАЛА: FCM (приложение) > web-push. Телефон с приложением — устройство, которое
реально при пользователе; веб-подписка часто висит на десктопном браузере, который может быть
закрыт сутками. Плюс тап по пушу в приложении открывает нужный экран напрямую.

Механика — общий маркер «эпизода простоя». Оба канала считают эпизод одинаково:
пользователь простаивает, если MAX(user_words.last_seen) старше IDLE_HOURS, и напоминание за
этот эпизод ещё не слали — т.е. `last_reminded_at IS NULL OR last_reminded_at < last_active`
(ровно условие HAVING в webpush.reminder_loop).

  1. Тик FCM берёт юзеров, у которых есть живой FCM-токен и эпизод не закрыт.
  2. Если веб-канал успел ПЕРВЫМ (какая-то строка push_subscriptions уже помечена
     last_reminded_at >= last_active) — юзер пропускается, второго пуша не будет.
  3. Иначе тик СНАЧАЛА «занимает» эпизод: помечает last_reminded_at = now у веб-подписок
     юзера, и только ПОТОМ шлёт в FCM. Так web-воркер (он читает ту же колонку) этого юзера
     на своём проходе уже не увидит — гонка двух лупов не даёт дубля.
  4. Если ни один FCM-токен не доставился — захват откатывается (возвращаем прежние значения),
     и веб-канал пришлёт напоминание сам на следующем проходе. Худший случай при падении
     процесса между 3 и 4 — пропущенное напоминание, а не два пуша.
"""
import os
import json
import asyncio
from datetime import datetime, timedelta

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from config import logger
from db import core as db_core
from auth import get_current_user
# Политика напоминаний общая с веб-каналом — не дублируем её здесь, чтобы каналы не разъехались.
from webpush import IDLE_HOURS, CHECK_INTERVAL_SEC, REMINDER, _quiet_now

# Креденшелы — из окружения, как VAPID: либо содержимое service-account JSON целиком,
# либо путь к файлу. Ничего не задано → канал выключен.
FCM_SERVICE_ACCOUNT_JSON = os.getenv("FCM_SERVICE_ACCOUNT_JSON", "")
FCM_SERVICE_ACCOUNT_FILE = os.getenv("FCM_SERVICE_ACCOUNT_FILE", "")
FCM_PROJECT_ID = os.getenv("FCM_PROJECT_ID", "")  # необязательно: по умолчанию из JSON
FCM_SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
FCM_ENDPOINT = "https://fcm.googleapis.com/v1/projects/{project}/messages:send"

MAX_TOKEN_LEN = 4096          # реальные FCM-токены ~163 симв.; кап от мусора в теле запроса
ALLOWED_PLATFORMS = ("android", "ios", "web")


def configured():
    return bool(FCM_SERVICE_ACCOUNT_JSON or FCM_SERVICE_ACCOUNT_FILE)


# ---------- креденшелы / OAuth2 ----------
# Библиотеку не добавляли: google-auth УЖЕ в requirements (им auth.py проверяет Google ID-token),
# и он же умеет service-account → OAuth2 access token с автообновлением. Ручная подпись JWT была бы
# лишним самописным крипто-кодом при готовой зависимости.
_CREDS = None
_SA_INFO = None


def _sa_info():
    """dict сервис-аккаунта (кэш). Ошибки чтения/парсинга — наверх, ловит вызывающий."""
    global _SA_INFO
    if _SA_INFO is None:
        if FCM_SERVICE_ACCOUNT_JSON:
            _SA_INFO = json.loads(FCM_SERVICE_ACCOUNT_JSON)
        else:
            with open(FCM_SERVICE_ACCOUNT_FILE, "r", encoding="utf-8") as f:
                _SA_INFO = json.load(f)
    return _SA_INFO


def _project_id():
    return FCM_PROJECT_ID or _sa_info().get("project_id") or ""


def _access_token():
    """Свежий OAuth2-токен сервис-аккаунта (google-auth сам рефрешит по истечении)."""
    global _CREDS
    if _CREDS is None:
        from google.oauth2 import service_account
        _CREDS = service_account.Credentials.from_service_account_info(_sa_info(), scopes=[FCM_SCOPE])
    if not _CREDS.valid:
        from google.auth.transport.requests import Request
        _CREDS.refresh(Request())
    return _CREDS.token


# ---------- хранилище подписок (своя таблица, своё соединение — как в webpush) ----------
async def _exec(sql, params=()):
    # DATABASE_URL берём из db.core по месту, а не снимком на импорте: путь подменяется в тестах.
    async with aiosqlite.connect(db_core.DATABASE_URL) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(sql, params)
        rows = await cur.fetchall()
        await db.commit()
        return rows


async def save_token(user_id, token, platform="android"):
    token = (token or "").strip()
    if not token or len(token) > MAX_TOKEN_LEN:
        raise ValueError("bad token")
    platform = (platform or "android").strip().lower()
    if platform not in ALLOWED_PLATFORMS:
        platform = "android"
    now = datetime.utcnow().isoformat()
    # один токен = одно устройство: при смене аккаунта на телефоне владелец перепривязывается,
    # эпизод напоминания сбрасывается (новый юзер — своя история).
    await _exec(
        """INSERT INTO fcm_subscriptions (user_id, token, platform, last_reminded_at, created_at)
           VALUES (?, ?, ?, NULL, ?)
           ON CONFLICT(token) DO UPDATE SET user_id=excluded.user_id, platform=excluded.platform,
               last_reminded_at=NULL""",
        (user_id, token, platform, now),
    )


async def delete_token(token, user_id):
    # owner-scoped: юзер отписывает только своё устройство (иначе чужим токеном можно отписать другого)
    await _exec("DELETE FROM fcm_subscriptions WHERE token = ? AND user_id = ?", (token, user_id))


async def delete_token_any(token):
    # воркерная чистка мёртвого токена (UNREGISTERED/INVALID_ARGUMENT): владелец неважен
    await _exec("DELETE FROM fcm_subscriptions WHERE token = ?", (token,))


# ---------- отправка (FCM HTTP v1) ----------
def _is_dead_token(status_code, body):
    """Токен протух/невалиден → подписку удалить (аналог 404/410 у web-push).
    FCM v1: 404 NOT_FOUND + errorCode UNREGISTERED (приложение снесли/токен ротировали),
    400 INVALID_ARGUMENT (мусорный токен). Прочие коды (401/429/5xx) — временные, не трогаем."""
    if status_code not in (400, 404):
        return False
    err = (body or {}).get("error") or {}
    status = (err.get("status") or "").upper()
    codes = {(d.get("errorCode") or "").upper() for d in (err.get("details") or []) if isinstance(d, dict)}
    return status in ("NOT_FOUND", "INVALID_ARGUMENT") or bool(codes & {"UNREGISTERED", "INVALID_ARGUMENT"})


def _send_sync(token, payload):
    """Блокирующая отправка одного пуша. Возвращает (ok, gone). gone=True → подписку удалить."""
    try:
        import requests
    except Exception as e:  # пакета нет — канал молча выключен, но бэкенд жив
        logger.warning(f"fcm: requests недоступен: {e}")
        return (False, False)
    try:
        project = _project_id()
        if not project:
            logger.warning("fcm: project_id не определён — отправка пропущена")
            return (False, False)
        access = _access_token()
    except Exception as e:
        logger.warning(f"fcm: креденшелы недоступны: {str(e)[:160]}")
        return (False, False)
    message = {
        "message": {
            "token": token,
            "notification": {"title": payload.get("title", ""), "body": payload.get("body", "")},
            "data": {"url": str(payload.get("url", ""))},
            "android": {"priority": "high", "notification": {"click_action": "OPEN_APP"}},
        }
    }
    try:
        r = requests.post(
            FCM_ENDPOINT.format(project=project),
            headers={"Authorization": f"Bearer {access}", "Content-Type": "application/json; UTF-8"},
            json=message, timeout=20,
        )
        if r.status_code == 200:
            return (True, False)
        try:
            body = r.json()
        except Exception:
            body = {}
        gone = _is_dead_token(r.status_code, body)
        if not gone:
            logger.warning(f"fcm error {r.status_code}: {str(body)[:160]}")
        return (False, gone)
    except Exception as e:
        logger.warning(f"fcm send error: {str(e)[:160]}")
        return (False, False)


async def _send(token, payload):
    return await asyncio.to_thread(_send_sync, token, payload)


# ---------- воркер напоминаний с дедупом по юзеру ----------
async def _candidates(cutoff):
    """Юзеры с FCM-токенами, простаивающие дольше IDLE_HOURS, кому за этот эпизод ещё не слали."""
    return await _exec(
        """SELECT f.id AS fid, f.user_id, f.token, MAX(uw.last_seen) AS last_active
           FROM fcm_subscriptions f
           JOIN user_words uw ON uw.user_id = f.user_id
           GROUP BY f.id
           HAVING last_active IS NOT NULL
              AND last_active <= ?
              AND (f.last_reminded_at IS NULL OR f.last_reminded_at < last_active)""",
        (cutoff,),
    )


async def _claim_web(user_id, last_active, now):
    """Занять эпизод у веб-канала. Возвращает (claimed, prev) либо None, если веб уже отработал.

    prev — прежние значения last_reminded_at занятых строк (для отката, если FCM не доставился).
    """
    rows = await _exec("SELECT id, last_reminded_at FROM push_subscriptions WHERE user_id = ?", (user_id,))
    for r in rows:
        lr = r["last_reminded_at"]
        if lr is not None and lr >= last_active:
            return None  # веб-канал уже напомнил за этот эпизод — второй пуш не шлём
    prev = [(r["id"], r["last_reminded_at"]) for r in rows]
    if prev:
        await _exec(
            "UPDATE push_subscriptions SET last_reminded_at = ? WHERE user_id = ?", (now, user_id))
    return prev


async def _rollback_web(prev):
    for sid, lr in prev:
        await _exec("UPDATE push_subscriptions SET last_reminded_at = ? WHERE id = ?", (lr, sid))


async def reminder_tick():
    """Один проход рассылки. Возвращает число юзеров, которым ушло напоминание через FCM."""
    cutoff = (datetime.utcnow() - timedelta(hours=IDLE_HOURS)).isoformat()
    rows = await _candidates(cutoff)
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], []).append(r)
    sent_users = 0
    for uid, devices in by_user.items():
        now = datetime.utcnow().isoformat()
        last_active = max(d["last_active"] for d in devices)
        prev = await _claim_web(uid, last_active, now)
        if prev is None:
            continue  # дедуп: напоминание за этот эпизод уже ушло в веб
        ok_any = False
        for d in devices:   # у юзера может быть несколько устройств — напоминание одно, на каждое
            ok, gone = await _send(d["token"], REMINDER)
            if gone:
                await delete_token_any(d["token"])
            elif ok:
                ok_any = True
                await _exec("UPDATE fcm_subscriptions SET last_reminded_at = ? WHERE id = ?", (now, d["fid"]))
        if ok_any:
            sent_users += 1
        else:
            await _rollback_web(prev)  # FCM не доставился — отдаём эпизод обратно веб-каналу
    return sent_users


async def reminder_loop():
    await asyncio.sleep(45)  # чуть раньше веб-воркера: при обоих каналах приоритет у приложения
    if not configured():
        logger.info("fcm: сервис-аккаунт не задан — пуши приложения выключены")
    while True:
        try:
            if not configured():
                await asyncio.sleep(3600)
                continue
            if _quiet_now():
                await asyncio.sleep(CHECK_INTERVAL_SEC)
                continue
            n = await reminder_tick()
            if n:
                logger.info(f"fcm: разослано напоминаний — {n}")
        except Exception as e:
            logger.warning(f"fcm reminder_loop: {str(e)[:200]}")
        await asyncio.sleep(CHECK_INTERVAL_SEC)


# ---------- API ----------
router = APIRouter(prefix="/push/fcm", tags=["push"])


@router.post("/subscribe")
async def subscribe(body: dict, user=Depends(get_current_user)):
    try:
        await save_token(user["id"], body.get("token"), body.get("platform", "android"))
    except ValueError:
        raise HTTPException(status_code=400, detail="bad token")
    return {"ok": True}


@router.post("/unsubscribe")
async def unsubscribe(body: dict, user=Depends(get_current_user)):
    tok = body.get("token")
    if tok:
        await delete_token(tok, user["id"])
    return {"ok": True}


_WORKER = None


def _start_worker():
    """Старт воркера вешаем на сам роутер: main.py остаётся с одной строкой include_router,
    а если модуль не подключился — фоновой задачи просто нет (падать нечему).
    Идемпотентно: include_router прокидывает startup-хендлер И в on_startup приложения,
    И в смёрженный lifespan роутера — без гарда воркер поднимался бы дважды."""
    global _WORKER
    if _WORKER is not None and not _WORKER.done():
        return
    _WORKER = asyncio.create_task(reminder_loop())
    logger.info(f"fcm push reminders: {'ON' if configured() else 'OFF (no service account)'}")


router.add_event_handler("startup", _start_worker)
