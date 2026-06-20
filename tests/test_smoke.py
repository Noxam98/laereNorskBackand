import pytest
from db.learning import apply_result, learning_stats
from tests.conftest import seed_user, seed_word


async def test_harness_apply_result(fresh_db):
    uid, did = await seed_user()
    pid, dwid = await seed_word(did, "hus", "дом")
    r = await apply_result(uid, pid, True, elapsed=2.0, mode="choice")
    assert r["ok"] is True
    assert r["strength"] >= 0
    s = await learning_stats(uid)
    assert s["total"] == 1
