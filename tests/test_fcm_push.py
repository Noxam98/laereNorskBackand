"""FCM-канал пушей (Android-приложение) и его дедуп с веб-пушами.

Проверяем сквозь публичные точки модуля (эндпоинты + проход воркера), сеть не трогаем:
  (1) subscribe/unsubscribe: подписка пишется, перепривязка токена к другому юзеру, чужой
      токен отписать нельзя, мусорный токен → 400;
  (2) ДЕДУП: у юзера есть и веб-подписка, и приложение → ровно ОДНО напоминание, каким бы
      из каналов ни начали (проверяем реальным прогоном обоих воркеров);
  (3) недоставка в FCM откатывает захват эпизода — веб-канал напомнит сам (напоминание не теряется);
  (4) протухший токен (UNREGISTERED/INVALID_ARGUMENT) удаляется из подписок.
"""
import asyncio
import types
from datetime import datetime, timedelta

import pytest
from fastapi import HTTPException

import fcm
import webpush
from db.core import _conn, _release, _now
from tests.conftest import seed_user, seed_word


# ---------------- инфраструктура ----------------

async def _seed_idle_user(username="u", idle_hours=20, with_web=True, with_fcm=True, token="tok"):
    """Юзер, который простаивает idle_hours, с подписками нужных каналов. → (uid, pool_id)"""
    uid, did = await seed_user(username)
    pid, _ = await seed_word(did, f"ord_{username}", "слово")
    last_seen = (datetime.utcnow() - timedelta(hours=idle_hours)).isoformat()
    db = await _conn()
    try:
        await db.execute(
            "INSERT INTO user_words (user_id, pool_id, last_seen, created_at) VALUES (?,?,?,?)",
            (uid, pid, last_seen, _now()))
        if with_web:
            await db.execute(
                """INSERT INTO push_subscriptions (user_id, endpoint, p256dh, auth, last_reminded_at, created_at)
                   VALUES (?,?,?,?,NULL,?)""",
                (uid, f"https://fcm.googleapis.com/fcm/send/{username}", "p", "a", _now()))
        if with_fcm:
            await db.execute(
                """INSERT INTO fcm_subscriptions (user_id, token, platform, last_reminded_at, created_at)
                   VALUES (?,?, 'android', NULL, ?)""",
                (uid, token, _now()))
        await db.commit()
        return uid, pid
    finally:
        await _release(db)


class _OneIteration:
    """Прогоняет бесконечный луп ровно на одну итерацию: первый sleep — стартовая пауза,
    второй (в конце итерации) — обрывает CancelledError."""

    def __init__(self):
        self.n = 0

    async def sleep(self, _sec):
        self.n += 1
        if self.n >= 2:
            raise asyncio.CancelledError
        await asyncio.sleep(0)


async def _run_web_loop_once(monkeypatch, db_path):
    """Реальный webpush.reminder_loop, один проход. Возвращает список endpoint'ов, куда ушло."""
    calls = []

    async def fake_send(endpoint, p256dh, auth, payload):
        calls.append(endpoint)
        return (True, False)

    monkeypatch.setattr(webpush, "DATABASE_URL", db_path)      # webpush снял путь снимком на импорте
    monkeypatch.setattr(webpush, "configured", lambda: True)   # VAPID в тестах не задан
    monkeypatch.setattr(webpush, "_quiet_now", lambda: False)  # тест не должен зависеть от времени суток
    monkeypatch.setattr(webpush, "_send", fake_send)
    monkeypatch.setattr(webpush, "asyncio", types.SimpleNamespace(
        sleep=_OneIteration().sleep, to_thread=asyncio.to_thread))
    with pytest.raises(asyncio.CancelledError):
        await webpush.reminder_loop()
    return calls


def _mock_fcm_send(monkeypatch, result=(True, False)):
    """Мок отправки в FCM. Возвращает список токенов, на которые звали."""
    calls = []

    async def fake_send(token, payload):
        calls.append(token)
        return result

    monkeypatch.setattr(fcm, "_send", fake_send)
    return calls


async def _fetch(sql, params=()):
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            return await cur.fetchall()
    finally:
        await _release(db)


# ---------------- (1) подписки ----------------

async def test_subscribe_stores_token(fresh_db):
    uid, _ = await seed_user("a")
    res = await fcm.subscribe({"token": "abc123", "platform": "android"}, user={"id": uid})
    assert res == {"ok": True}
    rows = await _fetch("SELECT user_id, token, platform FROM fcm_subscriptions")
    assert [(r["user_id"], r["token"], r["platform"]) for r in rows] == [(uid, "abc123", "android")]


async def test_subscribe_rebinds_token_to_new_owner(fresh_db):
    """Телефон переехал на другой аккаунт: токен один (UNIQUE), владелец обновляется."""
    uid1, _ = await seed_user("a")
    uid2, _ = await seed_user("b")
    await fcm.subscribe({"token": "abc123"}, user={"id": uid1})
    await fcm.subscribe({"token": "abc123"}, user={"id": uid2})
    rows = await _fetch("SELECT user_id FROM fcm_subscriptions")
    assert len(rows) == 1 and rows[0]["user_id"] == uid2


async def test_subscribe_rejects_bad_token(fresh_db):
    uid, _ = await seed_user("a")
    with pytest.raises(HTTPException) as e:
        await fcm.subscribe({"token": "  "}, user={"id": uid})
    assert e.value.status_code == 400
    with pytest.raises(HTTPException):
        await fcm.subscribe({"token": "x" * (fcm.MAX_TOKEN_LEN + 1)}, user={"id": uid})
    assert await _fetch("SELECT 1 FROM fcm_subscriptions") == []


async def test_unsubscribe_is_owner_scoped(fresh_db):
    uid1, _ = await seed_user("a")
    uid2, _ = await seed_user("b")
    await fcm.subscribe({"token": "abc123"}, user={"id": uid1})
    # чужой токен отписать нельзя
    assert await fcm.unsubscribe({"token": "abc123"}, user={"id": uid2}) == {"ok": True}
    assert len(await _fetch("SELECT 1 FROM fcm_subscriptions")) == 1
    # свой — можно
    await fcm.unsubscribe({"token": "abc123"}, user={"id": uid1})
    assert await _fetch("SELECT 1 FROM fcm_subscriptions") == []


# ---------------- (2) дедуп ----------------

async def test_dedup_native_wins_web_stays_silent(fresh_db, monkeypatch):
    """Веб + приложение у одного юзера: FCM отработал первым → веб-воркер молчит. Итого один пуш."""
    uid, _ = await _seed_idle_user("dual")
    fcm_calls = _mock_fcm_send(monkeypatch)

    assert await fcm.reminder_tick() == 1
    assert fcm_calls == ["tok"]

    web_calls = await _run_web_loop_once(monkeypatch, fresh_db)
    assert web_calls == []          # ← главное: второго уведомления нет
    assert len(fcm_calls) == 1


async def test_dedup_web_first_then_fcm_skips(fresh_db, monkeypatch):
    """Обратный порядок: веб успел первым → FCM этого юзера уже не берёт."""
    uid, _ = await _seed_idle_user("dual")

    web_calls = await _run_web_loop_once(monkeypatch, fresh_db)
    assert len(web_calls) == 1

    fcm_calls = _mock_fcm_send(monkeypatch)
    assert await fcm.reminder_tick() == 0
    assert fcm_calls == []


async def test_native_only_user_gets_push(fresh_db, monkeypatch):
    """Без веб-подписки канал работает сам по себе (захватывать нечего)."""
    await _seed_idle_user("native", with_web=False)
    fcm_calls = _mock_fcm_send(monkeypatch)
    assert await fcm.reminder_tick() == 1
    assert fcm_calls == ["tok"]
    # повторный проход в том же эпизоде простоя не шлёт второй раз
    assert await fcm.reminder_tick() == 0
    assert len(fcm_calls) == 1


async def test_active_user_is_not_reminded(fresh_db, monkeypatch):
    await _seed_idle_user("fresh_user", idle_hours=1)
    fcm_calls = _mock_fcm_send(monkeypatch)
    assert await fcm.reminder_tick() == 0
    assert fcm_calls == []


# ---------------- (3) откат захвата ----------------

async def test_failed_fcm_delivery_returns_episode_to_web(fresh_db, monkeypatch):
    """FCM не доставился (временная ошибка) → захват эпизода откатывается, напоминание
    пришлёт веб-канал. Напоминание не теряется и дублем не становится."""
    await _seed_idle_user("dual")
    fcm_calls = _mock_fcm_send(monkeypatch, result=(False, False))
    assert await fcm.reminder_tick() == 0
    assert fcm_calls == ["tok"]

    rows = await _fetch("SELECT last_reminded_at FROM push_subscriptions")
    assert rows[0]["last_reminded_at"] is None   # захват снят

    web_calls = await _run_web_loop_once(monkeypatch, fresh_db)
    assert len(web_calls) == 1


# ---------------- (4) протухший токен ----------------

async def test_dead_token_subscription_removed(fresh_db, monkeypatch):
    await _seed_idle_user("dual")
    _mock_fcm_send(monkeypatch, result=(False, True))   # FCM ответил UNREGISTERED
    assert await fcm.reminder_tick() == 0
    assert await _fetch("SELECT 1 FROM fcm_subscriptions") == []
    # эпизод отдан веб-каналу — юзер всё равно получит напоминание
    rows = await _fetch("SELECT last_reminded_at FROM push_subscriptions")
    assert rows[0]["last_reminded_at"] is None


def test_send_sync_builds_http_v1_request(monkeypatch):
    """Реальная отправка (_send_sync), сеть подменена: адрес/заголовок/тело — формат FCM HTTP v1,
    404 UNREGISTERED классифицируется как мёртвый токен."""
    import requests
    captured = {}

    class _Resp:
        status_code = 404

        def json(self):
            return {"error": {"status": "NOT_FOUND", "details": [{"errorCode": "UNREGISTERED"}]}}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured.update(url=url, headers=headers, body=json)
        return _Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(fcm, "_project_id", lambda: "proj-1")
    monkeypatch.setattr(fcm, "_access_token", lambda: "ya29.tok")

    assert fcm._send_sync("dev-token", {"title": "t", "body": "b", "url": "/#/learning"}) == (False, True)
    assert captured["url"] == "https://fcm.googleapis.com/v1/projects/proj-1/messages:send"
    assert captured["headers"]["Authorization"] == "Bearer ya29.tok"   # OAuth2, а не legacy server key
    msg = captured["body"]["message"]
    assert msg["token"] == "dev-token"
    assert msg["notification"] == {"title": "t", "body": "b"}
    assert msg["data"]["url"] == "/#/learning"

    _Resp.status_code = 200
    _Resp.json = lambda self: {"name": "projects/proj-1/messages/1"}
    assert fcm._send_sync("dev-token", fcm.REMINDER) == (True, False)


def test_send_sync_without_credentials_is_noop(monkeypatch):
    """Нет креденшелов → отправка молча выключена (ошибка наружу не летит, токен не чистим)."""
    monkeypatch.setattr(fcm, "_project_id", lambda: "")
    assert fcm._send_sync("dev-token", fcm.REMINDER) == (False, False)


def test_is_dead_token_classification():
    unreg = {"error": {"status": "NOT_FOUND", "details": [{"errorCode": "UNREGISTERED"}]}}
    assert fcm._is_dead_token(404, unreg)
    assert fcm._is_dead_token(400, {"error": {"status": "INVALID_ARGUMENT"}})
    # временные ошибки токен не убивают
    assert not fcm._is_dead_token(401, {"error": {"status": "UNAUTHENTICATED"}})
    assert not fcm._is_dead_token(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
    assert not fcm._is_dead_token(503, {"error": {"status": "UNAVAILABLE"}})
    assert not fcm._is_dead_token(500, {})


async def test_worker_starts_once(monkeypatch):
    """include_router прокидывает startup-хендлер двумя путями (on_startup + смёрженный
    lifespan) — воркер обязан подняться ровно один раз."""
    starts = []

    async def fake_loop():
        starts.append(1)
        await asyncio.sleep(3600)

    monkeypatch.setattr(fcm, "reminder_loop", fake_loop)
    monkeypatch.setattr(fcm, "_WORKER", None)
    try:
        fcm._start_worker()
        fcm._start_worker()
        await asyncio.sleep(0)
        assert starts == [1]
    finally:
        if fcm._WORKER is not None:
            fcm._WORKER.cancel()


def test_router_is_wired_into_app():
    """Роутер подключён к приложению и воркер стартует сам (main.py — только include)."""
    import main
    included = [getattr(r, "original_router", None) for r in main.app.routes]
    assert fcm.router in included
    paths = {getattr(r, "path", None) for r in fcm.router.routes}
    assert paths == {"/push/fcm/subscribe", "/push/fcm/unsubscribe"}
    assert fcm._start_worker in main.app.router.on_startup
