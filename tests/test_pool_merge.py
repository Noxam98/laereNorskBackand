"""D5: merge_pool_words — автономная операция, мутирующая SRS-прогресс юзера; раньше тест звал её
без единого assert и без реальных dict_words/user_words. Проверяем перепривязку прогресса на winner,
удаление loser и ветку UNIQUE-конфликта (когда у winner уже есть прогресс — прогресс loser теряется)."""
from db.pool_dedup import merge_pool_words
from db.learning import apply_result
from db.core import _conn, _release
from tests.conftest import seed_user, seed_word


async def _reps(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT reps FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as c:
            r = await c.fetchone()
            return r["reps"] if r else None
    finally:
        await _release(db)


async def _in_pool(pid):
    db = await _conn()
    try:
        async with db.execute("SELECT 1 FROM word_pool WHERE id=?", (pid,)) as c:
            return (await c.fetchone()) is not None
    finally:
        await _release(db)


async def test_merge_reattaches_progress(fresh_db):
    uid, did = await seed_user()
    winner, _ = await seed_word(did, "hund", "собака")
    loser, _ = await seed_word(did, "hunden", "собака (опр.)")
    await apply_result(uid, loser, True, mode="choice", direction="int2no")   # прогресс ТОЛЬКО на loser
    assert await _reps(uid, loser) is not None
    assert await _reps(uid, winner) is None

    assert await merge_pool_words(winner, loser) is True
    assert await _in_pool(loser) is False                 # loser удалён из пула
    assert await _reps(uid, loser) is None                # его user_words отвязаны
    assert await _reps(uid, winner) is not None           # прогресс переехал на winner


async def test_merge_unique_conflict_keeps_winner(fresh_db):
    uid, did = await seed_user()
    winner, _ = await seed_word(did, "hund", "собака")
    loser, _ = await seed_word(did, "hunden", "собака (опр.)")
    # у ОБОИХ есть прогресс → UPDATE OR IGNORE на winner конфликтует по UNIQUE(user_id,pool_id)
    await apply_result(uid, winner, True, mode="choice", direction="int2no")
    await apply_result(uid, loser, True, mode="choice", direction="int2no")

    assert await merge_pool_words(winner, loser) is True
    assert await _in_pool(loser) is False
    assert await _reps(uid, loser) is None                # прогресс loser отброшен (документируем поведение)
    assert await _reps(uid, winner) is not None           # прогресс winner цел


async def test_merge_noop_same_id(fresh_db):
    uid, did = await seed_user()
    w, _ = await seed_word(did, "hund", "собака")
    assert await merge_pool_words(w, w) is False           # winner==loser — no-op, слово цело
    assert await _in_pool(w) is True
