"""Целостность экзаменов-ворот и аудита (§2.4):

#1 ввод (type='input') сверяется ТОЛЬКО с норвежскими формами — показанный юзеру перевод-подсказку
   больше нельзя набрать и получить «верно»; типы-выбор (no2int/int2no/cloze) по-прежнему принимают
   перевод (варианты без подписей, сверка типонезависима);
#3 порог сдачи ворот масштабируется под реальный размер выборки (_pass_threshold), а не фикс PASS —
   иначе при < SAMPLE построенных вопросов ворота были непроходимы и приток новых заперт навсегда;
#4 аудит грейдит ТОЛЬКО реально выданный набор (просроченные ≤ AUDIT_CAP) — клиент не может двигать
   audit_due / де-сертифицировать сертифицированное слово, которое ему не выдавали.
"""
import json
import pytest
from datetime import datetime, timedelta
from db.exams import _exam_answer_ok, _pass_threshold, _question_buildable
from db.learning import (
    apply_result, build_gate_exam, grade_gate_exam, build_audit, grade_audit,
    PACK_FIRST, SAMPLE, PASS, AUDIT_CAP, FIRST_AUDIT_DAYS,
)
from db.core import _conn, _release
from tests.conftest import seed_user, seed_word


# ---------------- helpers ----------------

async def _master(uid, pid):
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    return await apply_result(uid, pid, True, mode="input", direction="int2no")


async def _seed_mastered_pack(uid, did, n, prefix="w"):
    out = []
    for i in range(n):
        no, ru = f"{prefix}{i}", f"пер{i}"
        pid, _ = await seed_word(did, no, ru)
        await _master(uid, pid)
        out.append((pid, no, ru))
    return out


async def _seed_certified_pack(uid, did, n, prefix="w"):
    out = await _seed_mastered_pack(uid, did, n, prefix)
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


async def _strip_translation(pid, no):
    """Убрать перевод у слова пула → слово перестаёт быть «построимым» в вопрос (для #3)."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET data=? WHERE id=?",
                         (json.dumps({"translate": {"no": [no]}, "part_of_speech": "noun"}), pid))
        await db.commit()
    finally:
        await _release(db)


def _row_for(no="hus", ru="дом", forms=None):
    return {"pool_id": 1, "norwegian": no,
            "data": json.dumps({"translate": {"ru": [ru]}}),
            "forms": json.dumps(forms) if forms else None}


# ---------------- #1: ввод не засчитывает показанный перевод ----------------

def test_input_rejects_translation_accepts_norwegian_unit():
    r = _row_for(no="hus", ru="дом")
    # ввод (accept_translation=False): показанный перевод НЕ принимается, норв. слово — принимается
    assert _exam_answer_ok(r, "ru", "дом", accept_translation=False) is False
    assert _exam_answer_ok(r, "ru", "hus", accept_translation=False) is True
    # выбор (accept_translation=True, дефолт): и перевод, и норвежское принимаются (типонезависимо)
    assert _exam_answer_ok(r, "ru", "дом", accept_translation=True) is True
    assert _exam_answer_ok(r, "ru", "hus", accept_translation=True) is True
    assert _exam_answer_ok(r, "ru", "дом") is True   # дефолт — как выбор (обратная совместимость)


def test_input_accepts_wordform_not_translation_unit():
    # словоформа норвежского принимается на вводе, перевод — нет
    r = _row_for(no="hus", ru="дом", forms={"def_sg": "huset", "pos": "noun"})
    assert _exam_answer_ok(r, "ru", "huset", accept_translation=False) is True
    assert _exam_answer_ok(r, "ru", "дом", accept_translation=False) is False


async def test_gate_input_type_rejects_shown_translation(fresh_db):
    """type='input' + ответ = показанный перевод → НЕ засчитывается → ворота не сдаются."""
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    answers = [{"pool_id": pid, "answer": ru, "type": "input"} for pid, no, ru in pack[:SAMPLE]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is False


async def test_gate_input_type_accepts_norwegian(fresh_db):
    """type='input' + ответ = норвежское слово → засчитывается → ворота сдаются."""
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    answers = [{"pool_id": pid, "answer": no, "type": "input"} for pid, no, ru in pack[:SAMPLE]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True


async def test_gate_choice_type_still_accepts_translation(fresh_db):
    """Регрессия: типы-выбор по-прежнему принимают перевод как ответ (грейд выбора не сломан)."""
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    answers = [{"pool_id": pid, "answer": ru, "type": "no2int"} for pid, no, ru in pack[:SAMPLE]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True


# ---------------- #3: порог масштабируется под размер выборки ----------------

def test_pass_threshold_scales_unit():
    assert _pass_threshold(SAMPLE) == PASS          # полная выборка (30) → штатный PASS (27)
    assert _pass_threshold(20) == 18                # ceil(0.9*20)
    assert _pass_threshold(10) == 9
    assert _pass_threshold(1) == 1
    assert _pass_threshold(0) == 1                  # пустой экзамен недостижим (correct=0 < 1)
    assert _pass_threshold(SAMPLE + 100) == PASS    # не выше PASS


async def test_gate_passable_when_few_questions_buildable(fresh_db):
    """Если из пачки строится < SAMPLE вопросов, ворота ДОЛЖНЫ оставаться проходимыми:
    порог = ceil(0.9*len), а не фикс PASS=27 (иначе приток новых заперт навсегда)."""
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, PACK_FIRST)
    keep = pack[:10]
    for pid, no, ru in pack[10:]:      # у остальных убираем перевод → они не дают вопрос
        await _strip_translation(pid, no)
    ex = await build_gate_exam(uid, lang="ru")
    assert len(ex["questions"]) == 10                    # построилось ровно 10 вопросов
    assert ex["pass"] == 9                               # порог = ceil(0.9*10), НЕ 27
    # 9 верных из 10 построимых → сдача (при фикс-PASS=27 было бы навсегда невозможно)
    answers = [{"pool_id": pid, "answer": ru, "type": "no2int"} for pid, no, ru in keep[:9]]
    res = await grade_gate_exam(uid, answers, lang="ru")
    assert res["passed"] is True


async def test_question_buildable_matches_stripped(fresh_db):
    """_question_buildable совпадает с реальностью: слово без перевода не построимо, с переводом — да."""
    uid, did = await seed_user()
    pack = await _seed_mastered_pack(uid, did, 4)
    strip_pid, strip_no, _ = pack[0]
    await _strip_translation(strip_pid, strip_no)
    db = await _conn()
    try:
        from db.learning import _fetch_user_words
        rows = await _fetch_user_words(db, uid)
    finally:
        await _release(db)
    by = {r["pool_id"]: r for r in rows}
    assert _question_buildable(by[strip_pid], "ru", []) is False
    assert _question_buildable(by[pack[1][0]], "ru", []) is True


# ---------------- #4: аудит грейдит только выданный набор ----------------

async def test_audit_grades_only_served_due_set(fresh_db):
    """Клиент не может через grade_audit трогать НЕ выданное (не-due) сертифицированное слово:
    его верный ответ не «освежает» срок, а неверный не де-сертифицирует."""
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    due_pid, _, due_ru = pack[0]
    other_pid, other_no, other_ru = pack[1]   # сертиф., но audit_due в будущем → НЕ в выборке
    await _set_audit_due(uid, due_pid, (datetime.utcnow() - timedelta(days=1)).isoformat())

    res = await grade_audit(uid, [
        {"pool_id": other_pid, "answer": "__forget__"},   # попытка де-сертифицировать не-due слово
        {"pool_id": due_pid, "answer": due_ru},            # реально выданное — учитывается
    ], lang="ru")
    assert res["checked"] == 1                 # обработан ТОЛЬКО due_pid, other_pid проигнорирован
    assert res["refreshed"] == 1 and res["forgot"] == 0
    assert (await _row(uid, other_pid))["certified"] == 1   # чужое слово НЕ де-сертифицировано


async def test_audit_input_type_rejects_translation(fresh_db):
    """На аудите type='input' с показанным переводом → забыто (де-серт), с норвежским → освежено."""
    uid, did = await seed_user()
    pack = await _seed_certified_pack(uid, did, PACK_FIRST)
    pid, no, ru = pack[0]
    await _set_audit_due(uid, pid, (datetime.utcnow() - timedelta(days=1)).isoformat())
    res = await grade_audit(uid, [{"pool_id": pid, "answer": ru, "type": "input"}], lang="ru")
    assert res["checked"] == 1 and res["forgot"] == 1 and res["refreshed"] == 0
    assert (await _row(uid, pid))["certified"] == 0
