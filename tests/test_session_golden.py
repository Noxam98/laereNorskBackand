"""ЭТАП 0: golden-слепок build_session на фиксированном сиде.

Структурное сравнение (не строки!), фикстура специально содержит СВЯЗКИ (tie)
в ключах сортировки: две новые карточки с одинаковым level/freq — стабильность
сортировки (единственный незадокументированный инвариант из карт) проверяется явно.
Слепок обновляется ОСОЗНАННО: если этап рефактора меняет порядок/состав — это сигнал
разобраться, а не молча перезаписать."""
import json

from db.core import _conn, _release
from tests.conftest import seed_user

from db.learning import build_session


async def _seed_word(dbc, did, uid, no, ru, *, uw=None, freq=3.0):
    cur = await dbc.execute(
        "INSERT INTO word_pool (norwegian, data, pos, level, freq, created_at) "
        "VALUES (?,?,?,?,?,datetime('now'))",
        (no, json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": "noun"}),
         "noun", "A1", freq))
    pid = cur.lastrowid
    await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                      (did, pid))
    if uw:
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, strength, reps, correct, incorrect, "
            "interval_days, due_at, modes, created_at) "
            f"VALUES (?,?,?,?,?,?,?,{uw['due']},?,datetime('now'))",
            (uid, pid, uw["strength"], uw["reps"], uw["correct"], uw["incorrect"],
             uw["interval"], json.dumps(uw.get("modes", {}))))
    return pid


async def _fixture(uid, did):
    dbc = await _conn()
    try:
        pids = {}
        # СВЯЗКА: два свежих слова с ИДЕНТИЧНЫМИ level/freq — tie в сортировке новых
        pids["nyA"] = await _seed_word(dbc, did, uid, "nyA", "новоеА", freq=3.5)
        pids["nyB"] = await _seed_word(dbc, did, uid, "nyB", "новоеБ", freq=3.5)
        # due-повтор (просрочен)
        pids["due1"] = await _seed_word(dbc, did, uid, "due1", "повтор", uw={
            "strength": 60, "reps": 4, "correct": 4, "incorrect": 0, "interval": 2,
            "due": "datetime('now','-1 day')", "modes": {"choice_int2no": "1"}})
        # слабое (2+ ошибок, низкая сила)
        pids["weak1"] = await _seed_word(dbc, did, uid, "weak1", "слабое", uw={
            "strength": 15, "reps": 5, "correct": 2, "incorrect": 3, "interval": 1,
            "due": "datetime('now','+1 day')", "modes": {}})
        await dbc.commit()
        return pids
    finally:
        await _release(dbc)


def _shape(res, pids):
    byname = {v: k for k, v in pids.items()}
    return {
        "comp": {k: res["composition"][k] for k in
                 ("fresh", "review", "weak", "progress", "phrases", "grammar", "phase", "total")},
        "seq": [(byname.get(w["pool_id"], "?"), w.get("mode"), w.get("step")) for w in res["words"]],
    }


async def test_build_session_golden(fresh_db):
    uid, did = await seed_user("golden")
    pids = await _fixture(uid, did)
    res = await build_session(uid, size=10)
    got = _shape(res, pids)
    # детерминизм: повторный вызов тем же состоянием — тот же слепок
    res2 = await build_session(uid, size=10)
    assert _shape(res2, pids) == got, "build_session недетерминирован на одном состоянии"

    golden = GOLDEN
    assert got == golden, f"\nСлепок изменился!\nбыло:  {golden}\nстало: {got}"


# Слепок текущего поведения (04.07.2026, до рефактора). Обновлять ОСОЗНАННО.
# Порядок seq фиксирует и стабильность сортировки на связке nyA/nyB (равные level/freq):
# из-за stable sort первым идёт вставленный раньше (nyA) — если tie-break сломается,
# слепок это покажет.
GOLDEN = {
    "comp": {"fresh": 2, "review": 0, "weak": 1, "progress": 1,
             "phrases": 0, "grammar": 0, "phase": "words", "total": 4},
    "seq": [
        ("due1", "build", "build_int2no"),    # просроченный повтор — первым
        ("weak1", "choice", "choice_int2no"),  # слабое — следом
        ("nyA", "study", "card"),              # новые карточки — в хвосте, стабильный порядок
        ("nyB", "study", "card"),
    ],
}
