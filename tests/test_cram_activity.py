"""Режим ЗАУЧИВАНИЯ пишет только дневной журнал (цель/стрик/точность) и НЕ трогает SRS.

Дрилл гоняет слова набора по кругу до чистого прогона — если бы он кормил /learning/answer,
зубрёжка двигала бы клетки рампы и интервалы повторений. Поэтому у него свой вход:
note_activity (POST /learning/activity).
"""
from db.core import _conn, _release
from db.learning import note_activity, learning_stats, apply_result, status_of
from tests.conftest import seed_user, seed_word


async def _row(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return dict(r) if r else None


async def test_activity_counts_toward_daily_goal(fresh_db):
    uid, _did = await seed_user()
    assert (await learning_stats(uid))["today"]["done"] == 0
    await note_activity(uid, 12, 9)
    st = await learning_stats(uid)
    assert st["today"]["done"] == 12
    assert st["streak"] == 1                      # день активный → стрик пошёл
    assert st["accuracy"] == 75                   # 9/12


async def test_activity_accumulates_within_day(fresh_db):
    uid, _did = await seed_user()
    await note_activity(uid, 5, 5)
    await note_activity(uid, 3, 1)
    assert (await learning_stats(uid))["today"]["done"] == 8


async def test_activity_does_not_touch_srs(fresh_db):
    """Главный инвариант: рампа и расписание слова после заучивания те же."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    before = await _row(uid, pid)
    await note_activity(uid, 40, 30)
    after = await _row(uid, pid)
    for f in ("modes", "due_at", "interval_days", "ease", "strength", "reps", "correct", "incorrect", "mastered"):
        assert before[f] == after[f], f
    assert status_of(after, {}) == status_of(before, {})


async def test_activity_clamped(fresh_db):
    """Журнал кормит стрик и рейтинг — накрутку одним запросом клампим."""
    uid, _did = await seed_user()
    r = await note_activity(uid, 10_000, 10_000)
    assert r["answers"] == 500 and r["correct"] == 500
    assert (await learning_stats(uid))["today"]["done"] == 500


async def test_activity_correct_never_exceeds_answers(fresh_db):
    uid, _did = await seed_user()
    r = await note_activity(uid, 4, 99)
    assert r == {"ok": True, "answers": 4, "correct": 4}


async def test_activity_zero_is_noop(fresh_db):
    """Выход из заучивания без единого ответа не должен создавать «активный день» (ложный стрик)."""
    uid, _did = await seed_user()
    await note_activity(uid, 0, 0)
    st = await learning_stats(uid)
    assert st["today"]["done"] == 0 and st["streak"] == 0
