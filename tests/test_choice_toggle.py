"""Ступень «выбор из вариантов» отключается юзером (gamePrefs.choiceStage, дефолт ВКЛ).

ВЫКЛ → клетки выбора не выдаются (слово идёт сразу на продукцию: сборка/ввод, у фраз — порядок
слов), а сдача более сложной ступени их ЗАСЧИТЫВАЕТ. Ключевой инвариант: «выучено»/CEFR/пачка
экзамена считаются по ПОЛНОЙ рампе — тумблер их не двигает ни туда, ни обратно (грандфатеринг
в обе стороны). Аудио-клетка остаётся под своим тумблером: выключить выбор ≠ отменить слух.
"""
import json
from db.core import _conn, _release
from db.learning import (
    build_session, build_listen_session, listen_status, apply_result, required_cells,
    status_of, REQUIRED_CELLS, PHRASE_CELLS, _AUDIO_CELL, _next_step,
)
from db.users import set_user_game_prefs, get_user_choice
from srs import cells, steps
from srs.cells import CONTENT, PHRASE, FUNC_CHOICE
from tests.conftest import seed_user, seed_word


# ── чистое ядро: какие клетки выдаём и какие засчитываем ─────────────────────

def test_ramp_cells_drops_choice_for_content():
    """Выбор ВЫКЛ + аудио ВЫКЛ: у обычного слова остаётся продукция (сборка → ввод)."""
    assert cells.ramp_cells(CONTENT) == REQUIRED_CELLS                       # тумблер вкл — как было
    assert cells.ramp_cells(CONTENT, choice_on=False) == ("build_int2no", "input_int2no")


def test_ramp_cells_keeps_audio_cell_when_audio_on():
    """Аудио ВКЛ: choice_no2int — не «выбор вариантов», а слуховая партия (свой тумблер).
    Тумблер выбора её не снимает, иначе слово выучивалось бы мимо сдачи на слух."""
    eff = cells.ramp_cells(CONTENT, choice_on=False, audio_on=True)
    assert eff == (_AUDIO_CELL, "build_int2no", "input_int2no")
    assert cells.skipped_choice(CONTENT, choice_on=False, audio_on=True) == ("choice_int2no",)
    assert cells.skipped_choice(CONTENT, choice_on=False) == ("choice_int2no", _AUDIO_CELL)
    assert cells.skipped_choice(CONTENT) == ()                              # тумблер вкл — ничего не пропускаем


def test_ramp_cells_phrase_and_function_word():
    """У фраз выбор перевода снимается (остаются порядок слов → сборка → ввод). У служебных без
    cloze рампа СОСТОИТ из выбора — её тумблер не трогает, иначе слово нечем сдать."""
    assert cells.ramp_cells(PHRASE, choice_on=False) == PHRASE_CELLS[1:]
    assert cells.ramp_cells(FUNC_CHOICE, choice_on=False) == cells.cells_of(FUNC_CHOICE)
    assert cells.skipped_choice(FUNC_CHOICE, choice_on=False) == ()


def test_full_ramp_is_not_affected():
    """cells_of/required_cells — ПОЛНАЯ рампа: по ней считается «выучено», тумблер её не меняет."""
    row = {"norwegian": "hus", "data": json.dumps({"part_of_speech": "noun"})}
    assert required_cells(row) == REQUIRED_CELLS == cells.cells_of(CONTENT)


def test_next_step_skips_choice():
    """Совсем новое → карточка; дальше сразу сборка (ступень выбора выключена)."""
    assert steps.next_step(CONTENT, {}, attempts=0, audio_on=True, choice_on=False) == ("card", "study", None)
    assert steps.next_step(CONTENT, {}, attempts=1, audio_on=True, choice_on=False) == \
        ("build_int2no", "build", "int2no")
    assert steps.next_step(CONTENT, {}, attempts=1, audio_on=True) == \
        ("choice_int2no", "choice", "int2no")          # тумблер вкл — прежний порядок


def test_next_step_anti_deadlock():
    """Битое состояние (продукция сдана, выбор пуст) при ВКЛЮЧЁННОМ тумблере не подвешивает слово:
    второй проход идёт по полной рампе и выдаёт клетку выбора, а не None навсегда."""
    modes = {"build_int2no": "1", "input_int2no": "1", _AUDIO_CELL: "1"}
    assert steps.next_step(CONTENT, modes, attempts=5, audio_on=False, choice_on=False) == \
        ("choice_int2no", "choice", "int2no")


# ── интеграция: зачёт пропущенных клеток и «выучено» ─────────────────────────

async def _modes(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return (dict(r) if r else {}), (json.loads(r["modes"]) if r and r["modes"] else {})


async def _choice_off(uid, **extra):
    await set_user_game_prefs(uid, json.dumps({"choiceStage": False, **extra}))
    assert await get_user_choice(uid) is False


async def test_session_gives_build_instead_of_choice(fresh_db):
    """Тумблер ВЫКЛ → в дневной сессии слово приходит на сборку, а не на выбор."""
    uid, did = await seed_user()
    await _choice_off(uid)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="study")          # карточка-знакомство (attempts > 0)
    res = await build_session(uid, size=20)
    w = [x for x in res["words"] if x["pool_id"] == pid]
    assert w and w[0]["step"] == "build_int2no"


async def test_production_credits_skipped_choice(fresh_db):
    """Сдача сборки засчитывает пропущенную клетку выбора (продукция сильнее узнавания).
    Аудио-клетку НЕ засчитывает: при аудио ВКЛ её закрывает только слуховая партия."""
    uid, did = await seed_user()
    await _choice_off(uid)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    _row, m = await _modes(uid, pid)
    assert m.get("choice_int2no") == "1"                      # зачтено сборкой
    assert m.get(_AUDIO_CELL, "") != "1"                      # слух не зачитываем


async def test_mastery_after_listen_with_choice_off(fresh_db):
    """Аудио ВКЛ + выбор ВЫКЛ: сборка → ввод → слово ЖДЁТ СЛУХ (не выучено), и только сдача
    на слух закрывает рампу. Выключение выбора не должно открывать обход слуховой партии."""
    uid, did = await seed_user()
    await _choice_off(uid)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, True, mode="input", direction="int2no")
    row, m = await _modes(uid, pid)
    assert status_of(row, m) != "mastered"
    assert (await listen_status(uid))["pending"] == 1
    listen = await build_listen_session(uid, size=20)
    assert pid in {w["pool_id"] for w in listen["words"]}
    await apply_result(uid, pid, True, mode="choice", direction="no2int")   # сдал слух
    row, m = await _modes(uid, pid)
    assert all(m.get(c) == "1" for c in REQUIRED_CELLS) and status_of(row, m) == "mastered"


async def test_audio_off_full_ramp_by_production(fresh_db):
    """Аудио ВЫКЛ + выбор ВЫКЛ: сборка → ввод, и слово выучено — обе клетки выбора зачтены."""
    uid, did = await seed_user()
    await _choice_off(uid, audio=False)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, True, mode="input", direction="int2no")
    row, m = await _modes(uid, pid)
    assert all(m.get(c) == "1" for c in REQUIRED_CELLS)
    assert status_of(row, m) == "mastered" and row["mastered"] == 1


async def test_grandfathering_back_and_forth(fresh_db):
    """Возврат тумблера не «разучивает» пачкой: слово, доведённое при выключённом выборе,
    остаётся выученным (клетки зачтены реально), и обратно — тоже."""
    uid, did = await seed_user()
    await _choice_off(uid, audio=False)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, True, mode="input", direction="int2no")
    await set_user_game_prefs(uid, json.dumps({"choiceStage": True, "audio": False}))   # вернули выбор
    row, m = await _modes(uid, pid)
    assert status_of(row, m) == "mastered"
    assert _next_step(row, m, audio_on=False, choice_on=True) is None


async def test_error_rollback_respects_disabled_choice(fresh_db):
    """Ошибка на сборке при выключённом выборе не откатывает на клетку, которую не выдают
    (ниже сборки ступеней нет), а ошибка на вводе откатывает на сборку — как обычно."""
    uid, did = await seed_user()
    await _choice_off(uid, audio=False)
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, False, mode="build", direction="int2no")
    _row, m = await _modes(uid, pid)
    assert m.get("build_int2no") == "" and m.get("choice_int2no") == "1"   # зачтённый выбор цел
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, False, mode="input", direction="int2no")
    _row, m = await _modes(uid, pid)
    assert m.get("input_int2no") == "" and m.get("build_int2no") == ""      # штатный откат на ступень назад
    assert m.get("choice_int2no") == "1"


async def test_function_word_keeps_choice_ramp(fresh_db):
    """Служебное слово (рампа «только выбор») тумблером не ломается — оно по-прежнему выдаётся."""
    uid, did = await seed_user()
    await _choice_off(uid)
    pid, _ = await seed_word(did, "og", "и", pos="conjunction")
    row = {"norwegian": "og", "data": json.dumps({"part_of_speech": "conjunction"}),
           "correct": 1, "incorrect": 0}
    assert _next_step(row, {}, audio_on=True, choice_on=False) == ("choice_int2no", "choice", "int2no")
    assert pid  # слово заведено (гейт ввода служебных — отдельная логика, см. test_cloze_bank)
