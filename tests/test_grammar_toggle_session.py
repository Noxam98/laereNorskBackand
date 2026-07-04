"""ЭТАП 0: характеризация инцидента #2 — тумблер грамматики не должен оголять сессию.

Два РАЗНЫХ механизма (критика плана требует развести): трек форм (learning_forms,
form_srs) и грамм-overlay (learning_grammar, местоимения). В обоих случаях при
grammar=False сессия ветерана либо непуста, либо честно объясняет причину."""
import json

from db.core import _conn, _release
from tests.conftest import seed_user

from db.learning import build_session
from db.users import set_user_game_prefs


async def set_user_prefs(uid, prefs: dict):
    await set_user_game_prefs(uid, json.dumps(prefs))


RAMP_DONE = json.dumps({c: "1" for c in
                        ("choice_int2no", "choice_no2int", "build_int2no", "input_int2no")})


async def _seed_mastered_formable(uid, did, no="stol"):
    """Выученное сущ. с формами: кандидат трека форм (несданные клетки)."""
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, forms, created_at) "
            "VALUES (?,?,?,?,?,datetime('now'))",
            (no, json.dumps({"translate": {"no": [no], "ru": ["стул"]},
                             "part_of_speech": "noun"}), "noun", "A1",
             json.dumps({"pos": "noun", "gender": "en", "def_sg": f"{no}en",
                         "indef_pl": f"{no}er", "def_pl": f"{no}ene"})))
        pid = cur.lastrowid
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, strength, reps, correct, incorrect, "
            "interval_days, due_at, modes, created_at) "
            "VALUES (?,?,?,?,?,?,?,datetime('now','+3 days'),?,datetime('now'))",
            (uid, pid, 80, 8, 8, 0, 3, RAMP_DONE))
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


async def test_grammar_off_forms_track_not_black_hole(fresh_db):
    """Ветеран: всё mastered, работа есть только у ТРЕКА ФОРМ. grammar=False:
    сессия не обязана дать формы (выключены!), но обязана быть непустой ЛИБО
    честно назвать причину пустоты — не молчаливый words:[]"""
    uid, did = await seed_user("vet_forms")
    await _seed_mastered_formable(uid, did)
    await set_user_prefs(uid, {"grammar": False})
    res = await build_session(uid, size=10)
    comp = res["composition"]
    assert comp["total"] > 0 or comp.get("reason"), comp
    # формы при выключенном тумблере не подмешиваются
    assert comp.get("grammar", 0) == 0, comp


async def test_grammar_on_forms_track_serves_forms(fresh_db):
    """Контроль: тот же ветеран с grammar=True получает работу трека форм."""
    uid, did = await seed_user("vet_forms_on")
    await _seed_mastered_formable(uid, did)
    await set_user_prefs(uid, {"grammar": True})
    res = await build_session(uid, size=10)
    comp = res["composition"]
    assert comp["total"] > 0, comp


async def test_grammar_off_overlay_pronoun_case(fresh_db):
    """Второй механизм: грамм-overlay (местоимения, PRONOUN_PARADIGM, форм в БД нет).
    grammar=False → overlay не подмешивается, сессия не пустеет молча."""
    uid, did = await seed_user("vet_pron")
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) "
            "VALUES (?,?,?,?,datetime('now'))",
            ("min", json.dumps({"translate": {"no": ["min"], "ru": ["мой"]},
                                "part_of_speech": "pronoun"}), "pronoun", "A1"))
        pid = cur.lastrowid
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, strength, reps, correct, incorrect, "
            "interval_days, due_at, modes, created_at) "
            "VALUES (?,?,?,?,?,?,?,datetime('now','+3 days'),?,datetime('now'))",
            (uid, pid, 80, 8, 8, 0, 3, RAMP_DONE))
        await dbc.commit()
    finally:
        await _release(dbc)
    await set_user_prefs(uid, {"grammar": False})
    res = await build_session(uid, size=10)
    comp = res["composition"]
    assert comp["total"] > 0 or comp.get("reason"), comp
    assert comp.get("grammar", 0) == 0, comp
