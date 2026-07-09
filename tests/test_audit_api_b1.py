"""Регрессии аудита безопасности API «Учёбы» (полоса B1):

  (1) /learning/answer — IDOR: чужой pool_id → 404, своё слово → проходит;
  (2) /learning/report — дедуп: повторная жалоба того же юзера не крутит report_count;
  (3) зачёт no2int — эхо ВИДИМОГО норвежского промпта не засчитывается (обход ворот забывания);
  (4) placement — омоним грейдится по ТОЧНОЙ записи пула (carried pool_id), а не по get_pool_id(no);
  (5) битые элементы answers (не-dict) не роняют грейд (500 → игнор);
  (6) /learning/level — невалидный уровень → 422.

Стиль/фикстуры — как в остальных тестах: fresh_db + seed_user/seed_word, вызов функций напрямую.
"""
import json

import pytest
from fastapi import HTTPException

from db.core import _conn, _release, _now
from db import report_word, reported_count
from db.learning import apply_result, grade_gate_exam, grade_placement, grade_audit
from routers.learning import learning_answer_route, learning_level_route
from models import LearningAnswer, LevelBody
from tests.conftest import seed_user, seed_word


# ---------------- helpers ----------------

async def _pool_only_word(no, ru="перевод", pos="noun", level="A1"):
    """Слово ТОЛЬКО в общем пуле (без привязки к словарям юзера) → «чужое» для _owns_pool."""
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,?)",
            (no, data, pos, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _report_count_of(pid):
    db = await _conn()
    try:
        async with db.execute("SELECT COALESCE(report_count,0) rc FROM word_pool WHERE id=?", (pid,)) as cur:
            r = await cur.fetchone()
            return r["rc"] if r else None
    finally:
        await _release(db)


async def _master(uid, pid):
    """Довести слово до mastered (все 4 клетки рампы), НЕ сертифицируя → живёт в несданной пачке."""
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, True, mode="input", direction="int2no")


# ---------------- (1) /answer IDOR ----------------

async def test_answer_idor_foreign_pool_id_404(fresh_db):
    """Ответ по ЧУЖОМУ (никогда не изучавшемуся) pool_id → 404 (защита от накрутки SRS/рейтинга)."""
    uid, did = await seed_user()
    foreign = await _pool_only_word("fremmed", "чужое")   # в пуле, но не в словарях юзера
    with pytest.raises(HTTPException) as ei:
        await learning_answer_route(
            LearningAnswer(pool_id=foreign, correct=True, mode="study"), user={"id": uid})
    assert ei.value.status_code == 404


async def test_answer_owned_pool_id_passes(fresh_db):
    """Ответ по своему слову (в словаре юзера) проходит и возвращает состояние SRS."""
    uid, did = await seed_user()
    owned, _ = await seed_word(did, "hus", "дом")     # seed_word кладёт слово в словарь юзера
    res = await learning_answer_route(
        LearningAnswer(pool_id=owned, correct=True, mode="study"), user={"id": uid})
    assert isinstance(res, dict)


# ---------------- (2) /report дедуп ----------------

async def test_report_dedup_does_not_double_count(fresh_db):
    """Повторная жалоба того же юзера на то же слово → status 'already', report_count НЕ растёт."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "spamword", "спам")

    r1 = await report_word(pid, uid)
    assert r1["status"] == "queued"
    assert await _report_count_of(pid) == 1
    assert await reported_count() == 1

    r2 = await report_word(pid, uid)
    assert r2["status"] == "already"
    assert await _report_count_of(pid) == 1     # НЕ 2 — глобальный счётчик не накручен
    assert await reported_count() == 1


# ---------------- (3) зачёт: no2int эхо промпта — неверно ----------------

async def test_no2int_echo_of_prompt_graded_wrong(fresh_db):
    """no2int: норвежская лемма (hus) — ВИДИМЫЙ вопрос. Эхо промпта не должно «сдавать» ворота —
    иначе клиент всегда проходит forgetting-gate, набирая показанное слово."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    await _master(uid, pid)                     # mastered, не сертифицировано → в пачке
    res = await grade_gate_exam(uid, [{"pool_id": pid, "answer": "hus", "type": "no2int"}], lang="ru")
    assert res["passed"] is False
    # без фикса перевод-независимая сверка приняла бы норв. словоформу «hus» → ворота бы сдались


# ---------------- (4) placement: омоним по carried pool_id ----------------

async def test_placement_homograph_graded_by_carried_pool_id(fresh_db):
    """Омонимы (одно написание, разные смыслы/переводы): грейд по carried pool_id смысла B,
    а не по get_pool_id(no) без pos (тот вернул бы старший id = смысл A → верный ответ «мимо»)."""
    uid, did = await seed_user()
    a_id = await _pool_only_word("ligge", "смыслA", pos="verb", level="B1")   # старший id
    b_id = await _pool_only_word("ligge", "смыслB", pos="noun", level="B1")

    # вопрос-выбор (B1) несёт pool_id смысла B; ответ = перевод смысла B
    ans = [{"no": "ligge", "level": "B1", "type": "choice", "answer": "смыслB", "pool_id": b_id}]
    res = await grade_placement(uid, "ru", ans)
    assert res["perLevel"]["B1"]["ok"] == 1      # засчитано по смыслу B, а не по A (get_pool_id→a_id)


# ---------------- (5) битые answers не роняют грейд ----------------

async def test_malformed_answers_do_not_500(fresh_db):
    """Не-dict элементы в answers (строка/число/None) должны игнорироваться, а не ронять грейд (500)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")

    # placement: мусор игнорируется, валидный ответ всё ещё считается
    res = await grade_placement(
        uid, "ru", ["строка", 123, None, {"no": "hus", "level": "A1", "answer": "дом"}])
    assert isinstance(res, dict) and "perLevel" in res

    # зачётный экзамен: только мусор → не падаем, ворота не сдаются
    res2 = await grade_gate_exam(uid, ["строка", None, 5], lang="ru")
    assert isinstance(res2, dict) and res2.get("passed") is False

    # аудит: то же
    res3 = await grade_audit(uid, ["строка", None, {"answer": 5}], lang="ru")
    assert isinstance(res3, dict) and res3.get("checked") == 0


# ---------------- (6) /level валидация ----------------

async def test_level_route_rejects_invalid_level(fresh_db):
    uid, _ = await seed_user()
    with pytest.raises(HTTPException) as ei:
        await learning_level_route(LevelBody(level="ZZ"), user={"id": uid})
    assert ei.value.status_code == 422


async def test_level_route_accepts_valid_level(fresh_db):
    uid, _ = await seed_user()
    res = await learning_level_route(LevelBody(level="A1"), user={"id": uid})
    assert res["ok"] is True and res["level"] == "A1"
