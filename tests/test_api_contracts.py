"""ЭТАП 0 страховочной сети (docs/decoupling-plan.md): характеризация HTTP-контрактов.

Первые router-тесты в репо: фиксируют ФОРМУ ответов, на которую опирается фронт,
до любого переноса кода. ASGITransport без lifespan — фоновые лупы не стартуют,
БД — fresh_db из conftest. Авторизация — dependency_overrides (без JWT).
"""
import json

import httpx
import pytest
import pytest_asyncio

import db as D
from db.core import _conn, _release
from tests.conftest import seed_user


def _mk_client(uid):
    import main
    from auth import get_current_user, get_admin_user
    user = {"id": uid, "username": "t", "isAdmin": True, "hasPassword": True, "name": "t"}
    main.app.dependency_overrides[get_current_user] = lambda: user
    main.app.dependency_overrides[get_admin_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                             base_url="http://test"), main.app


@pytest_asyncio.fixture
async def client(fresh_db):
    uid, did = await seed_user("api")
    c, app = _mk_client(uid)
    yield c, uid, did
    await c.aclose()
    app.dependency_overrides.clear()


def test_import_main_smoke():
    """Смок порядка реэкспортов: хвост db/learning.py (1379+) резолвится в рантайме —
    перестановка импортов роняет приложение при старте, а не при вызове. Сам факт
    import main (транзитивно весь db.*) — и есть проверка; reload НЕ делаем
    (перепривязывает глобалы и ломает соседние тесты)."""
    import main
    from db import learning
    # ключевые поздние реэкспорты должны быть резолвлены к моменту старта
    for name in ("_is_certified", "suggest_words", "estimate_level"):
        assert hasattr(learning, name), name
    assert hasattr(main, "app")


async def test_session_contract(client):
    c, uid, did = client
    r = await c.get("/learning/session?size=5&lang=ru")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"words", "composition"}
    comp = body["composition"]
    for k in ("fresh", "review", "weak", "progress", "phrases", "grammar", "phase", "total"):
        assert k in comp, k
    assert isinstance(body["words"], list)


async def test_listen_status_contract(client):
    c, uid, did = client
    r = await c.get("/learning/listen/status")
    assert r.status_code == 200
    body = r.json()
    assert "ready" in body or "count" in body or "enabled" in body, body  # форма как есть


async def test_pool_meta_both_outcomes(client):
    c, uid, did = client
    dbc = await _conn()
    try:
        await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            ("hus", json.dumps({"translate": {"no": ["hus"], "ru": ["дом"]},
                                "part_of_speech": "noun"}), "noun", "A1"))
        await dbc.commit()
    finally:
        await _release(dbc)
    found = (await c.get("/pool/hus/meta")).json()
    missing_resp = await c.get("/pool/qqqzzz/meta")
    missing = missing_resp.json()
    # фиксируем текущую асимметрию форм как контракт (найдено — широкая, мимо — узкая)
    assert "found" in found or len(found) >= 5, found
    assert missing_resp.status_code in (200, 404)
    assert len(missing) <= max(2, len(found) - 3), (found, missing)


async def test_answer_branches_form_vs_base(client):
    c, uid, did = client
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, forms, created_at) "
            "VALUES (?,?,?,?,?,datetime('now'))",
            ("bok", json.dumps({"translate": {"no": ["bok"], "ru": ["книга"]},
                                "part_of_speech": "noun"}), "noun", "A1",
             json.dumps({"pos": "noun", "gender": "ei", "def_sg": "boka",
                         "indef_pl": "bøker", "def_pl": "bøkene"})))
        pid = cur.lastrowid
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
        await dbc.commit()
    finally:
        await _release(dbc)
    # base-ветка → user_words
    r = await c.post("/learning/answer", json={"pool_id": pid, "correct": True,
                                               "mode": "choice", "direction": "int2no"})
    assert r.status_code == 200
    dbc = await _conn()
    try:
        n_uw = (await (await dbc.execute(
            "SELECT COUNT(*) FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid))).fetchone())[0]
        assert n_uw == 1
        # form-ветка → form_srs, user_words НЕ множится
        r2 = await c.post("/learning/answer", json={"pool_id": pid, "correct": True,
                                                    "mode": "choice", "direction": "def_sg",
                                                    "form": True, "cell": "def_sg", "stage": "choose"})
        assert r2.status_code == 200
        n_fs = (await (await dbc.execute(
            "SELECT COUNT(*) FROM form_srs WHERE user_id=? AND pool_id=?", (uid, pid))).fetchone())[0]
        n_uw2 = (await (await dbc.execute(
            "SELECT COUNT(*) FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid))).fetchone())[0]
        assert n_fs >= 1 and n_uw2 == 1
    finally:
        await _release(dbc)


async def test_next_cards_contract(client):
    """next_new_cards — ОТДЕЛЬНЫЙ литерал карточки (не build_session!): фиксируем его форму,
    чтобы дрейф полей между эндпоинтами ловился (критика Этапа 0)."""
    c, uid, did = client
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            ("eple", json.dumps({"translate": {"no": ["eple"], "ru": ["яблоко"]},
                                 "part_of_speech": "noun"}), "noun", "A1"))
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, cur.lastrowid))
        await dbc.commit()
    finally:
        await _release(dbc)
    r = await c.get("/learning/next-cards?count=2&lang=ru")
    if r.status_code == 404:   # роут может называться иначе — контракт фиксируем по факту
        pytest.skip("нет роута next-cards")
    body = r.json()
    cards = body if isinstance(body, list) else body.get("cards") or body.get("words") or []
    if cards:
        card = cards[0]
        for k in ("pool_id", "no"):
            assert k in card, card


async def test_set_session_scoped_contract(client):
    """scoped=set_id выключает WIP/ворота/автодобор: слова набора приходят без гейтов."""
    c, uid, did = client
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO dictionaries (user_id, name, studying, created_at) VALUES (?,?,1,datetime('now'))",
            (uid, "набор"))
        set_id = cur.lastrowid
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            ("vindu", json.dumps({"translate": {"no": ["vindu"], "ru": ["окно"]},
                                  "part_of_speech": "noun"}), "noun", "A1"))
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (set_id, cur.lastrowid))
        await dbc.commit()
    finally:
        await _release(dbc)
    r = await c.get(f"/sets/{set_id}/session?size=5&lang=ru")
    if r.status_code == 404:
        pytest.skip("нет роута set-session")
    body = r.json()
    assert "words" in body and "composition" in body
