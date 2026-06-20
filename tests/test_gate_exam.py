"""Зачётный экзамен-ворота (§2.4-A): несданная пачка mastered+certified=0, выборка/сдача,
демоут промахов ×2 при провале, сертификация всей пачки при сдаче, блок притока новых."""
import json
import pytest
from db.learning import (
    apply_result, gate_status, new_words_blocked, build_gate_exam, grade_gate_exam,
    suggest_words, status_of, PACK_FIRST, PACK, SAMPLE, PASS,
)
from db.core import _conn, _release
from tests.conftest import seed_user, seed_word


async def _master(uid, pid):
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    return await apply_result(uid, pid, True, mode="input", direction="int2no")


async def _seed_mastered_pack(uid, did, n, prefix="w"):
    """Создать n слов и довести каждое до mastered. Вернуть [(pid, no, ru)]."""
    out = []
    for i in range(n):
        no = f"{prefix}{i}"
        ru = f"пер{i}"
        pid, _ = await seed_word(did, no, ru)
        await _master(uid, pid)
        out.append((pid, no, ru))
    return out


async def _certified_count(uid):
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) AS n FROM user_words WHERE user_id=? AND certified=1", (uid,)) as cur:
            return (await cur.fetchone())["n"]
    finally:
        await _release(db)


async def _row(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            return dict(await cur.fetchone())
    finally:
        await _release(db)


async def test_gate_threshold_first_then_pack(fresh_db):
    uid, did = await seed_user()
    st = await gate_status(uid)
    assert st["threshold"] == PACK_FIRST and st["open"] is False
    # одно слово, сертифицируем вручную через сдачу — но сначала проверим переключение порога
    pid, _ = await seed_word(did, "hus", "дом")
    await _master(uid, pid)
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET certified=1 WHERE user_id=? AND pool_id=?", (uid, pid))
        await db.commit()
    finally:
        await _release(db)
    st = await gate_status(uid)
    assert st["threshold"] == PACK   # после первой сертификации порог = PACK


async def test_gate_opens_at_first_threshold(fresh_db):
    uid, did = await seed_user()
    await _seed_mastered_pack(uid, did, PACK_FIRST)
    st = await gate_status(uid)
    assert st["pack"] == PACK_FIRST
    assert st["open"] is True
    assert await new_words_blocked(uid) is True


async def test_gate_below_threshold_not_open(fresh_db):
    uid, did = await seed_user()
    await _seed_mastered_pack(uid, did, PACK_FIRST - 1)
    st = await gate_status(uid)
    assert st["open"] is False
    assert await new_words_blocked(uid) is False


async def test_build_gate_exam_shape(fresh_db):
    uid, did = await seed_user()
    await _seed_mastered_pack(uid, did, PACK_FIRST)
    ex = await build_gate_exam(uid, lang="ru")
    assert len(ex["questions"]) == SAMPLE
    for q in ex["questions"]:
        assert set(q.keys()) >= {"no", "pool_id", "options"}
        assert len(q["options"]) == 4


async def test_gate_pass_certifies_whole_pack(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    # отвечаем верно на SAMPLE первых слов пачки
    answers = [{"pool_id": pid, "answer": ru} for pid, no, ru in pack[:SAMPLE]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True
    # вся пачка (PACK_FIRST) сертифицирована, не только отвеченные
    assert await _certified_count(uid) == PACK_FIRST
    st = await gate_status(uid)
    assert st["pack"] == 0 and st["open"] is False
    assert await new_words_blocked(uid) is False


async def test_gate_fail_demotes_double(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    # SAMPLE вопросов: сделаем ровно столько промахов, чтобы не добрать PASS.
    # верных = PASS-1 → промахов среди отвеченных = SAMPLE-(PASS-1)
    n_correct = PASS - 1
    answers = []
    for i in range(SAMPLE):
        pid, no, ru = pack[i]
        ans = ru if i < n_correct else "__неверно__"
        answers.append({"pool_id": pid, "answer": ans})
    miss = SAMPLE - n_correct
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is False
    assert res["demoted"] == miss * 2   # промахи + столько же самых слабых (штраф ×2)
    assert await _certified_count(uid) == 0
    # демоутнутые ушли из пачки → pack уменьшился на demoted
    st = await gate_status(uid)
    assert st["pack"] == PACK_FIRST - miss * 2
    # конкретный промах демоутнут (mastered → review): клетки рампы сброшены, certified=0
    pid0 = pack[n_correct][0]
    row = await _row(uid, pid0)
    modes = json.loads(row["modes"] or "{}")
    assert all(modes.get(c, "") != "1" for c in ["choice_no2int", "input_int2no"])
    assert row["certified"] == 0
    assert status_of(row, modes) != "mastered"


async def test_suggest_blocked_when_gate_open(fresh_db):
    uid, did = await seed_user()
    await _seed_mastered_pack(uid, did, PACK_FIRST)
    res = await suggest_words(uid, count=5)
    assert res.get("blocked") is True
    assert res["added"] == 0
