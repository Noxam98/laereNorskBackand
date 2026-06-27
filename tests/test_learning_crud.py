"""CRUD/запросы «Учёбы» вне основного движка сессии: добавление/удаление слова,
ручная смена статуса (know/known/reset/unarchive), очередь Smart Review (get_due),
дневная активность, поиск по слову (get_learning q — покрывает вложенный hit())."""
from db.learning import (
    learning_add, learning_remove, set_status, get_due, get_activity,
    get_learning, build_session, apply_result, _due_str,
)
from db.core import _conn, _release
from db.pool import get_or_create_pool
from tests.conftest import seed_user, seed_word


def _ids(res):
    return [w["pool_id"] for w in res["words"]]


async def test_learning_add_idempotent_and_remove(fresh_db):
    uid, _did = await seed_user()
    pid = await get_or_create_pool("eple", {
        "word": "eple", "translate": {"ru": ["яблоко"]}, "part_of_speech": "noun", "level": "A1"})
    r = await learning_add(uid, pid)
    assert r.get("ok") and not r.get("duplicate")
    assert pid in _ids(await get_learning(uid))
    assert (await learning_add(uid, pid)).get("duplicate")     # повторное добавление идемпотентно
    await learning_remove(uid, pid)
    assert pid not in _ids(await get_learning(uid))            # ушло из ротации


async def test_remove_archives_then_readd_restores(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "stol", "стул")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")   # есть прогресс
    await learning_remove(uid, pid)
    assert pid not in _ids(await get_learning(uid))
    await learning_add(uid, pid)                               # вернули — прогресс восстановлен
    assert pid in _ids(await get_learning(uid))


async def test_set_status_known_and_reset(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    await set_status(uid, pid, "known")
    w = next(w for w in (await get_learning(uid))["words"] if w["pool_id"] == pid)
    assert w["status"] == "mastered"
    await set_status(uid, pid, "reset")
    w = next(w for w in (await get_learning(uid))["words"] if w["pool_id"] == pid)
    assert w["status"] == "new"


async def test_set_status_know_archives_and_unarchive(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bord", "стол")
    await set_status(uid, pid, "know")                         # в архив
    assert pid not in _ids(await build_session(uid, size=50))  # архивные не в сессии
    await set_status(uid, pid, "unarchive")
    assert pid in _ids(await get_learning(uid))                # вернулось в ротацию


async def test_get_due_returns_overdue(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "katt", "кошка")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET due_at=? WHERE user_id=? AND pool_id=?", (_due_str(-3), uid, pid))
        await db.commit()
    finally:
        await _release(db)
    assert pid in _ids(await get_due(uid, 20))


async def test_get_activity_empty_user(fresh_db):
    uid, _did = await seed_user()
    act = await get_activity(uid)
    assert isinstance(act.get("days"), list)


async def test_get_learning_search_by_word(fresh_db):
    uid, did = await seed_user()
    await seed_word(did, "vann", "вода")
    await seed_word(did, "brod", "хлеб")
    nos = [w["no"] for w in (await get_learning(uid, q="vann"))["words"]]
    assert "vann" in nos and "brod" not in nos
