"""Аудит-экзамен забывания (§2.4-B): сертификация выставляет audit_due (now+FIRST_AUDIT_DAYS),
build_audit берёт самые просроченные сертифицированные, grade_audit двигает срок при успехе и
де-сертифицирует при провале; тормоз новых при доле забытых > THROTTLE."""
import json
import pytest
from datetime import datetime, timedelta
from db.learning import (
    apply_result, grade_gate_exam, build_audit, grade_audit, build_session, status_of,
    new_words_blocked, audit_throttled, suggest_words, learning_stats,
    PACK_FIRST, SAMPLE, FIRST_AUDIT_DAYS, AUDIT_CAP, THROTTLE,
)
from db.core import _conn, _release, _now
from tests.conftest import seed_user, seed_word


async def _master(uid, pid):
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    return await apply_result(uid, pid, True, mode="input", direction="int2no")


async def _seed_certified_pack(uid, did, n, prefix="w"):
    """n слов → mastered → сдать зачётный экзамен → сертифицировано (audit_due проставлен)."""
    out = []
    for i in range(n):
        no, ru = f"{prefix}{i}", f"пер{i}"
        pid, _ = await seed_word(did, no, ru)
        await _master(uid, pid)
        out.append((pid, no, ru))
    answers = [{"pool_id": pid, "answer": ru} for pid, no, ru in out[:SAMPLE]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True
    return out


async def _row(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            return dict(await cur.fetchone())
    finally:
        await _release(db)


async def _set_audit_due(uid, pid, iso):
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET audit_due=? WHERE user_id=? AND pool_id=?", (iso, uid, pid))
        await db.commit()
    finally:
        await _release(db)


async def test_certification_sets_audit_due(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    pid = pack[0][0]
    row = await _row(uid, pid)
    assert row["certified"] == 1
    assert row["audit_due"]
    # ~ now + FIRST_AUDIT_DAYS (в будущем, поэтому ещё не на аудите)
    due = datetime.fromisoformat(row["audit_due"])
    expect = datetime.utcnow() + timedelta(days=FIRST_AUDIT_DAYS)
    assert abs((due - expect).total_seconds()) < 3600


async def test_build_audit_empty_when_not_due(fresh_db):
    uid, did = await seed_user()
    await _seed_certified_pack(uid, did, PACK_FIRST)
    # audit_due в будущем → никого не берём
    ex = await build_audit(uid, lang="ru")
    assert ex["questions"] == []


async def test_build_audit_picks_overdue_capped_and_ordered(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    # сделаем AUDIT_CAP+5 слов просроченными, с разными audit_due (давность убывает с индексом)
    overdue = pack[:AUDIT_CAP + 5]
    base = datetime.utcnow()
    for k, (pid, no, ru) in enumerate(overdue):
        # k=0 — самый давний (дольше всех ждёт)
        await _set_audit_due(uid, pid, (base - timedelta(days=100 - k)).isoformat())
    ex = await build_audit(uid, cap=AUDIT_CAP, lang="ru")
    assert len(ex["questions"]) == AUDIT_CAP   # потолок
    # первым — самый давний (pid из overdue[0])
    assert ex["questions"][0]["pool_id"] == overdue[0][0]
    for q in ex["questions"]:
        assert set(q.keys()) >= {"no", "pool_id", "options"}
        assert len(q["options"]) == 4


async def test_build_audit_ignores_non_certified(fresh_db):
    """Аудит берёт ТОЛЬКО сертифицированные слова: mastered, но несертифицированное
    слово с просроченным audit_due на аудит не попадает (его проверят ворота, не аудит)."""
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    cert_pid, _, cert_ru = pack[0]
    # сертифицированному ставим просрочку — он должен попасть
    await _set_audit_due(uid, cert_pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
    # отдельное mastered, но НЕсертифицированное слово с просроченным audit_due — мимо аудита
    pid2, _ = await seed_word(did, "lonewolf", "одиночка")
    await _master(uid, pid2)
    await _set_audit_due(uid, pid2, (datetime.utcnow() - timedelta(days=50)).isoformat())
    row2 = await _row(uid, pid2)
    assert row2["certified"] == 0   # ворота он не сдавал
    ex = await build_audit(uid, cap=AUDIT_CAP, lang="ru")
    picked = {q["pool_id"] for q in ex["questions"]}
    assert cert_pid in picked
    assert pid2 not in picked


async def test_grade_audit_correct_pushes_due_further(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    pid, no, ru = pack[0]
    await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
    res = await grade_audit(uid, [{"pool_id": pid, "answer": ru}], lang="ru")
    assert res == {"checked": 1, "refreshed": 1, "forgot": 0, "throttle": False}
    row = await _row(uid, pid)
    assert row["certified"] == 1
    # новый срок — заметно в будущем (минимум удвоение от now)
    due = datetime.fromisoformat(row["audit_due"])
    assert due > datetime.utcnow() + timedelta(days=FIRST_AUDIT_DAYS * 2 - 1)


async def test_grade_audit_interval_grows_monotonically_across_cycles(fresh_db):
    """Срок до следующего аудита РАСТЁТ между успешными циклами при аудите ВОВРЕМЯ:
    30 → 60 → 120 → 240 …, а не залипает на 60. Каждый цикл слово делаем просроченным
    на 1 день (аудит вовремя, overdue≈0) и сдаём верно — интервал должен удваиваться."""
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    pid, no, ru = pack[0]

    expected = [60, 120, 240]   # после 30(старт) при аудите вовремя — удвоение каждый цикл
    prev_interval = None
    for exp in expected:
        # аудит вовремя: due просрочен лишь на 1 день (overdue≈0, не должно влиять на рост)
        await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
        res = await grade_audit(uid, [{"pool_id": pid, "answer": ru}], lang="ru")
        assert res["refreshed"] == 1 and res["forgot"] == 0
        row = await _row(uid, pid)
        interval = row["audit_interval"]
        assert round(interval) == exp, f"ожидали интервал {exp}, получили {interval}"
        # монотонный рост
        if prev_interval is not None:
            assert interval > prev_interval
        prev_interval = interval
        # и audit_due согласован с интервалом (now + interval)
        due = datetime.fromisoformat(row["audit_due"])
        assert abs((due - (datetime.utcnow() + timedelta(days=exp))).total_seconds()) < 3600


async def test_grade_audit_wrong_decertifies_and_returns(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    pid, no, ru = pack[0]
    await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
    res = await grade_audit(uid, [{"pool_id": pid, "answer": "__неверно__"}], lang="ru")
    assert res["checked"] == 1 and res["forgot"] == 1 and res["refreshed"] == 0
    row = await _row(uid, pid)
    modes = json.loads(row["modes"] or "{}")
    assert row["certified"] == 0
    assert row["audit_due"] is None
    # снова не-mastered и вернулось именно в изучение (review/learning/weak)
    st = status_of(row, modes)
    assert st != "mastered"
    assert st in ("review", "learning", "weak")
    # клетки рампы сброшены (де-мастеринг)
    assert all(modes.get(c, "") != "1" for c in ("choice_no2int", "input_int2no"))
    # вернулось в очередь изучения: due проставлен на ближайшее, lapses вырос
    assert row["due_at"] is not None
    assert (row["lapses"] or 0) >= 1
    # и реально снова попадает в программу занятия (на доучивание)
    sess = await build_session(uid, size=PACK_FIRST)
    assert pid in {w["pool_id"] for w in sess["words"]}


async def test_grade_audit_throttle_when_many_forgot(fresh_db):
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    answers = []
    n = 10
    forgot = 0
    for i in range(n):
        pid, no, ru = pack[i]
        await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
        # забываем больше доли THROTTLE
        wrong = i < int(THROTTLE * n) + 2
        answers.append({"pool_id": pid, "answer": "__нет__" if wrong else ru})
        if wrong:
            forgot += 1
    res = await grade_audit(uid, answers, lang="ru")
    assert res["checked"] == n
    assert res["forgot"] == forgot
    assert res["throttle"] is True
    assert (forgot / n) > THROTTLE
    # тормоз РЕАЛЬНО применён на бэкенде: персистится и закрывает приток новых слов
    assert await audit_throttled(uid) is True
    assert await new_words_blocked(uid) is True
    # «Докинуть» новых заблокировано тормозом
    sug = await suggest_words(uid, count=5)
    assert sug["added"] == 0 and sug.get("blocked") is True
    # build_session не вводит новые (0 попыток) слова, пока тормоз активен
    pid_new, _ = await seed_word(did, "helt_nytt", "совсем_новое")
    sess = await build_session(uid, size=AUDIT_CAP + 10)
    assert pid_new not in {w["pool_id"] for w in sess["words"]}
    # статистика отражает тормоз
    stats = await learning_stats(uid)
    assert stats["audit"]["throttled"] is True


async def test_grade_audit_no_throttle_keeps_new_flowing(fresh_db):
    """Мало забытого (≤ THROTTLE) → тормоз не включается, приток новых открыт."""
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    # все аудиты сданы верно → forgot=0
    answers = []
    for i in range(10):
        pid, no, ru = pack[i]
        await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
        answers.append({"pool_id": pid, "answer": ru})
    res = await grade_audit(uid, answers, lang="ru")
    assert res["forgot"] == 0 and res["throttle"] is False
    assert await audit_throttled(uid) is False
    # после сертификации ворота уже сданы (had_cert), пачка пуста → новые открыты
    assert await new_words_blocked(uid) is False
