"""№2: выученное (mastered), НЕ сертифицированное слово, ставшее due → приходит на ПОВТОР
на стадии ввода (input_int2no). Верный ввод → остаётся mastered (интервал растёт). Неверный →
input+build сбрасываются (откат в рампу)."""
import pytest
from db.learning import build_session, apply_result, _due_str, REQUIRED_CELLS, status_of
from db.core import _conn, _release
from tests.conftest import seed_user, seed_word  # noqa


async def _set_due(pool_id, user_id, due_at):
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET due_at = ? WHERE pool_id = ? AND user_id = ?",
                         (due_at, pool_id, user_id))
        await db.commit()
    finally:
        await _release(db)


async def _master(uid, pid):
    """Прогнать слово по всей рампе верными ответами → mastered (все REQUIRED_CELLS='1')."""
    for cell in REQUIRED_CELLS:
        mode, direction = cell.split("_", 1)
        await apply_result(uid, pid, True, mode=mode, direction=direction)


async def _row(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as c:
            r = await c.fetchone()
        return dict(r) if r else None
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_due_mastered_comes_back_at_input(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hjelpe", "помогать", pos="verb")
    await _master(uid, pid)
    import json
    r = await _row(uid, pid)
    assert status_of(r, json.loads(r["modes"] or "{}")) == "mastered"
    assert not r.get("certified")
    # сделать due (в прошлом)
    await _set_due(pid, uid, _due_str(-3))

    res = await build_session(uid, size=20)
    word = next((w for w in res["words"] if w["pool_id"] == pid), None)
    assert word is not None, "выученное due-слово должно вернуться в сессию"
    assert word["mode"] == "input" and word["direction"] == "int2no", "повтор — на стадии ввода"
    assert word["step"] == "input_int2no"
    assert word["repeat"] is True, "повтор-бейдж"


@pytest.mark.asyncio
async def test_not_due_mastered_excluded(fresh_db):
    """Выученное, но ещё НЕ due — в обычную сессию не приходит (ждёт своего due)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "lese", "читать", pos="verb")
    await _master(uid, pid)
    await _set_due(pid, uid, _due_str(30))  # due в будущем
    res = await build_session(uid, size=20)
    assert all(w["pool_id"] != pid for w in res["words"]), "не-due выученное не должно приходить"


@pytest.mark.asyncio
async def test_wrong_input_rolls_back(fresh_db):
    """Неверный ввод на повторе → input и build сброшены → слово больше не mastered (откат)."""
    import json
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "skrive", "писать", pos="verb")
    await _master(uid, pid)
    # неверный ответ на стадии ввода
    await apply_result(uid, pid, False, mode="input", direction="int2no")
    r = await _row(uid, pid)
    modes = json.loads(r["modes"] or "{}")
    assert modes.get("input_int2no") != "1", "input сброшен"
    assert modes.get("build_int2no") != "1", "build откатан"
    assert status_of(r, modes) != "mastered", "после ошибки слово больше не выучено"
