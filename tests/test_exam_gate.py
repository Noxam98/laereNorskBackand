"""Exam gate (§2.4-A): несданная пачка = mastered & certified=0, ворота закрывают приток
новых при pack>=threshold, сдача (≥PASS) сертифицирует всю пачку, провал демоутит ×2 промаха.
Тесты проверяют реальное поведение SRS-слоя, а не подгоняются под реализацию."""
import json
import pytest
from db.learning import (
    apply_result, gate_status, new_words_blocked, grade_gate_exam,
    status_of, PACK_FIRST, PACK, SAMPLE, PASS,
)
from db.core import _conn, _release
from tests.conftest import seed_user, seed_word


async def _master(uid, pid):
    """Провести слово по всей рампе из 4 клеток → mastered."""
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    return await apply_result(uid, pid, True, mode="input", direction="int2no")


async def _seed_mastered_pack(uid, did, n, prefix="w"):
    """Создать n слов и довести каждое до mastered. Вернуть [(pid, no, ru)]."""
    out = []
    for i in range(n):
        no, ru = f"{prefix}{i}", f"пер{i}"
        pid, _ = await seed_word(did, no, ru)
        await _master(uid, pid)
        out.append((pid, no, ru))
    return out


async def _set_certified(uid, pids):
    db = await _conn()
    try:
        marks = ",".join("?" for _ in pids)
        await db.execute(
            f"UPDATE user_words SET certified=1 WHERE user_id=? AND pool_id IN ({marks})",
            [uid] + list(pids))
        await db.commit()
    finally:
        await _release(db)


async def _row(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            return dict(await cur.fetchone())
    finally:
        await _release(db)


# (1) пачка считает только mastered & certified=0
async def test_pack_counts_only_mastered_uncertified(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, 5)
    # ещё несколько слов, которые НЕ должны попасть в пачку:
    # — не-mastered (только пара клеток рампы)
    pid_partial, _ = await seed_word(did, "partial", "частично")
    await apply_result(uid, pid_partial, True, mode="choice", direction="no2int")
    await apply_result(uid, pid_partial, True, mode="choice", direction="int2no")
    # — совсем новое (0 попыток)
    await seed_word(did, "fresh", "новое")

    st = await gate_status(uid)
    assert st["pack"] == 5   # только 5 mastered, всё certified=0

    # сертифицируем 2 из пачки → они выпадают из счётчика
    await _set_certified(uid, [pack[0][0], pack[1][0]])
    st = await gate_status(uid)
    assert st["pack"] == 3   # mastered, но certified=1 в пачку не идут


# (2) при pack>=threshold ворота открыты → new_words_blocked True
async def test_gate_blocks_new_words_at_threshold(fresh_db):
    uid, did = await seed_user()
    # ниже порога — приток новых открыт
    await _seed_mastered_pack(uid, did, PACK_FIRST - 1)
    assert (await gate_status(uid))["open"] is False
    assert await new_words_blocked(uid) is False
    # ровно на пороге — ворота закрывают приток новых
    await _seed_mastered_pack(uid, did, 1, prefix="x")
    st = await gate_status(uid)
    assert st["pack"] == PACK_FIRST and st["threshold"] == PACK_FIRST
    assert st["open"] is True
    assert await new_words_blocked(uid) is True


# (3) grade с >=27 верных сертифицирует всю пачку и снимает блок
async def test_pass_certifies_whole_pack_and_unblocks(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    assert await new_words_blocked(uid) is True   # до экзамена ворота закрыты

    # ровно PASS верных ответов хватает на сдачу
    answers = [{"pool_id": pid, "answer": ru} for pid, no, ru in pack[:PASS]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True

    # сертифицирована ВСЯ пачка, включая неотвеченные слова
    for pid, no, ru in pack:
        assert (await _row(uid, pid))["certified"] == 1
    st = await gate_status(uid)
    assert st["pack"] == 0 and st["open"] is False
    assert await new_words_blocked(uid) is False   # порог теперь PACK, пачка пуста


# (4) провал демоутит промахи + равное число слабейших (всего 2×промаха)
async def test_fail_demotes_double_unmasters_them(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    # пометим часть пачки слабее, чтобы «самые слабые» были детерминированы и НЕ совпадали
    # с промахами: первые слова пачки — самые слабые по strength.
    db = await _conn()
    try:
        for i, (pid, no, ru) in enumerate(pack):
            await db.execute("UPDATE user_words SET strength=? WHERE user_id=? AND pool_id=?",
                             (i, uid, pid))
        await db.commit()
    finally:
        await _release(db)

    # отвечаем на SAMPLE слов; промахи делаем на слабейших нельзя (они самые слабые и так),
    # поэтому промахнёмся на сильных хвостовых, чтобы штрафные «слабейшие» отличались.
    n_correct = PASS - 1
    miss = SAMPLE - n_correct
    answered = pack[-SAMPLE:]          # хвост = самые сильные (strength больше)
    missed_pids = []
    answers = []
    for i, (pid, no, ru) in enumerate(answered):
        if i < n_correct:
            answers.append({"pool_id": pid, "answer": ru})
        else:
            answers.append({"pool_id": pid, "answer": "__неверно__"})
            missed_pids.append(pid)

    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is False
    assert res["demoted"] == miss * 2          # промахи + равное число слабейших

    # промахи перестали быть mastered и certified=0
    for pid in missed_pids:
        row = await _row(uid, pid)
        modes = json.loads(row["modes"] or "{}")
        assert status_of(row, modes) != "mastered"
        assert row["certified"] == 0

    # штрафные — самые слабые слова пачки (минимальные strength), не входящие в промахи
    weakest_pids = [pid for pid, no, ru in pack[:miss]]   # strength 0..miss-1
    for pid in weakest_pids:
        row = await _row(uid, pid)
        modes = json.loads(row["modes"] or "{}")
        assert status_of(row, modes) != "mastered"

    # пачка уменьшилась ровно на 2×промаха (демоутнутые больше не mastered&uncertified)
    st = await gate_status(uid)
    assert st["pack"] == PACK_FIRST - miss * 2
