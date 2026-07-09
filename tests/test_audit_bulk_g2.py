"""Аудит G2 — модерация bulk/import-путей и валидация тел списков.

Покрываем два инварианта, введённых security-фиксом:
  (a) bulk-импорт/генерация слов через ИИ кладут слово в ЛИЧНОЕ расширение автора
      (approved=0, created_by=user), а НЕ в общую Базу (approved=1) — иначе любой юзер
      инжектит произвольные слова всем в Базу и в дистракторы викторин;
  (b) POST /sets/membership с нечисловым элементом в pool_ids → 422 (а не 500 в SQL).

LLM замокан (ask_json), как в tests/test_autofill.py. Роут-тест — ASGITransport без lifespan
(фоновые лупы не стартуют), авторизация через dependency_overrides.
"""
import json

import httpx
import pytest
import pytest_asyncio

from db.core import _conn, _release
from tests.conftest import seed_user


@pytest.fixture
def mock_ask(monkeypatch):
    """Подменить ask_json фиксированным ответом модели в autofill_wordgen (там живут
    words_from_list / generate_set_words — они и зовут get_or_create_pool)."""
    import autofill_wordgen

    def _set(resp):
        async def fake(system, user, schema, **kw):
            return resp
        monkeypatch.setattr(autofill_wordgen, "ask_json", fake)
    return _set


async def _pool_row(pool_id):
    """approved/created_by записи пула по id (сырьё БД — проверяем реальное состояние, не ответ API)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT approved, created_by FROM word_pool WHERE id = ?", (pool_id,)
        ) as cur:
            r = await cur.fetchone()
            return dict(r) if r else None
    finally:
        await _release(db)


# ---------------- (a) bulk-пути → личное расширение автора (approved=0) ----------------

async def test_words_from_list_lands_pending_for_author(fresh_db, mock_ask):
    """Импорт списка слов (/import-words → words_from_list) кладёт новое слово как approved=0
    + created_by=user, а не в общую Базу."""
    import autofill_wordgen
    uid, _did = await seed_user("importer")
    mock_ask({"words": [{"word": "bil", "translate": {"ru": ["машина"]},
                         "part_of_speech": "noun", "level": "A1"}]})

    pids = await autofill_wordgen.words_from_list(["bil"], "ru", created_by=uid)
    assert len(pids) == 1
    row = await _pool_row(pids[0])
    assert row is not None
    assert row["approved"] == 0, "юзер-импорт не должен попадать в общую Базу (approved=1)"
    assert row["created_by"] == uid, "автор должен видеть своё pending-слово (created_by)"


async def test_generate_set_words_lands_pending_for_author(fresh_db, mock_ask):
    """AI-генерация набора (/sets/{id}/generate → generate_set_words) — тоже approved=0/created_by."""
    import autofill_wordgen
    uid, _did = await seed_user("gen")
    mock_ask({"words": [{"word": "hund", "translate": {"ru": ["собака"]},
                         "part_of_speech": "noun", "level": "A1"}]})

    pids = await autofill_wordgen.generate_set_words("животные", "A1", 5, "ru", created_by=uid)
    assert len(pids) == 1
    row = await _pool_row(pids[0])
    assert row is not None
    assert row["approved"] == 0
    assert row["created_by"] == uid


async def test_persist_default_stays_approved(fresh_db, mock_ask):
    """Регресс: дефолтный вызов (фоновые лупы, без created_by/approved) сохраняет ПРЕЖНЕЕ
    поведение — approved=1 (общая База). Не ломаем не-юзерский путь."""
    import autofill_wordgen
    items = [{"word": "katt", "translate": {"ru": ["кошка"]}, "part_of_speech": "noun", "level": "A1"}]
    pids = await autofill_wordgen._persist_word_items(items, 1)
    assert len(pids) == 1
    row = await _pool_row(pids[0])
    assert row["approved"] == 1 and row["created_by"] is None


# ---------------- (b) /sets/membership: нечисловой элемент → 422, не 500 ----------------

def _mk_client(uid):
    import main
    from auth import get_current_user
    user = {"id": uid, "username": "t", "isAdmin": False, "hasPassword": True, "name": "t"}
    main.app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                             base_url="http://test"), main.app


@pytest_asyncio.fixture
async def client(fresh_db):
    uid, did = await seed_user("member")
    c, app = _mk_client(uid)
    yield c, uid, did
    await c.aclose()
    app.dependency_overrides.clear()


async def test_membership_non_int_element_is_422(client):
    c, uid, did = client
    r = await c.post("/sets/membership", json={"pool_ids": [1, "abc", 3]})
    assert r.status_code == 422, r.text            # нечисловой элемент → 422, а НЕ 500 из SQL


async def test_membership_valid_ids_ok(client):
    c, uid, did = client
    r = await c.post("/sets/membership", json={"pool_ids": [1, 2, 3]})
    assert r.status_code == 200, r.text            # валидные int-и — как раньше (карта, пусть пустая)
    assert isinstance(r.json(), dict)


async def test_membership_non_list_is_422(client):
    c, uid, did = client
    r = await c.post("/sets/membership", json={"pool_ids": 5})
    assert r.status_code == 422, r.text            # pool_ids не список → 422, не 500 (len(int))
