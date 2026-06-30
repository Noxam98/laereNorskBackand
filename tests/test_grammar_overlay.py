"""Грамматический overlay (тир ★, NOUN-срез): отдельный слой ПОВЕРХ base-рампы — род (choice_gender)
и нерегулярное мн.ч. (input_indefpl) для уже ВЫУЧЕННЫХ существительных. Проверяем, что:
  (1) _grammar_cells даёт клетки по части речи + формам (нерегулярное мн.ч. → input, регулярное — нет);
  (2) overlay подмешивается в сессию только для mastered-слов с формами и гейтится тумблером профиля;
  (3) apply_result фиксирует грамм-клетку, но НЕ трогает base-рампу и «выучено»/CEFR (отдельный слой).
"""
import json
from db.core import _conn, _release
from db.learning import (
    build_session, apply_result, _is_mastered,
    _grammar_cells, _grammar_element, REQUIRED_CELLS, NOUN_GRAMMAR_CELLS, GRAMMAR_RATIO,
)
from db.users import set_user_game_prefs
from tests.conftest import seed_user, seed_word

_DATA_NOUN = json.dumps({"part_of_speech": "noun", "translate": {"ru": ["x"]}})
_BOK = {"pos": "noun", "gender": "ei", "def_sg": "boka", "indef_pl": "bøker", "def_pl": "bøkene"}
_BIL = {"pos": "noun", "gender": "en", "def_sg": "bilen", "indef_pl": "biler", "def_pl": "bilene"}


async def _set_forms(pool_id, forms):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?", (json.dumps(forms), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def _master(uid, pid):
    """Прогнать слово по всем 4 клеткам base-рампы → «выучено» (mastered)."""
    for cell in REQUIRED_CELLS:
        mode, direction = cell.split("_", 1)
        await apply_result(uid, pid, True, mode=mode, direction=direction)


# ── (1) _grammar_cells: по части речи + формам ──────────────────────────────────────────────
def test_grammar_cells_irregular_plural():
    # bøker ≠ предсказанного «boker» → мн.ч. нерегулярно → и род, и ввод мн.ч.
    assert _grammar_cells("bok", _DATA_NOUN, _BOK) == ["choice_gender", "input_indefpl"]


def test_grammar_cells_regular_plural_only_gender():
    # biler = предсказанное правилом → мн.ч. выводимо, проверять его не нужно → только род
    assert _grammar_cells("bil", _DATA_NOUN, _BIL) == ["choice_gender"]


def test_grammar_cells_non_noun_empty():
    assert _grammar_cells("snakke", json.dumps({"part_of_speech": "verb"}),
                          {"pos": "verb", "past": "snakket"}) == []


def test_grammar_cells_no_forms_empty():
    assert _grammar_cells("bok", _DATA_NOUN, None) == []
    assert _grammar_cells("bok", _DATA_NOUN, {}) == []


# ── _grammar_element: параметризованный контракт target/prompt/… ─────────────────────────────
def test_grammar_element_gender_shape():
    row = {"pool_id": 7, "norwegian": "bok", "mastered": 1}
    el = _grammar_element(row, "choice_gender", _BOK, {"translate": {"ru": ["книга"]}})
    assert el["mode"] == "choice" and el["direction"] == "gender" and el["grammar"] is True
    assert el["target"] == {"field": "gender", "value": "ei"}
    assert {o["w"] for o in el["options"]} == {"en", "ei", "et"}      # все три артикля
    assert set(el["distractors"]) == {"en", "et"}                     # кроме верного
    assert el["prompt"]["formLabel"] == "gender" and el["prompt"]["lemma"] == "bok"


def test_grammar_element_indefpl_shape():
    row = {"pool_id": 7, "norwegian": "bok", "mastered": 1}
    el = _grammar_element(row, "input_indefpl", _BOK, {})
    assert el["mode"] == "input" and el["direction"] == "indefpl"
    assert el["target"] == {"field": "indef_pl", "value": "bøker"}
    assert el["scoring"] == {"typoForgive": False}                    # формы вводим без прощения опечаток


# ── (2) overlay в сессии: только mastered + формы, гейт тумблером ────────────────────────────
async def test_overlay_surfaces_for_mastered_noun(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)
    res = await build_session(uid, size=20)
    grams = [w for w in res["words"] if w.get("grammar")]
    assert grams, "грамм-overlay должен подмешать упражнение для выученного сущ. с формами"
    g = grams[0]
    assert g["pool_id"] == pid and g["step"] in NOUN_GRAMMAR_CELLS
    assert res["composition"]["grammar"] == len(grams)


async def test_overlay_quota_proportional(fresh_db):
    """Грамм-квота = round(size*RATIO): много выученных сущ. с формами → не больше квоты."""
    uid, did = await seed_user()
    for i in range(15):
        pid, _ = await seed_word(did, f"bok{i}", f"книга{i}")
        await _set_forms(pid, _BOK)
        await _master(uid, pid)
    res = await build_session(uid, size=20)
    grams = [w for w in res["words"] if w.get("grammar")]
    assert len(grams) == round(20 * GRAMMAR_RATIO)


async def test_overlay_off_via_toggle(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)
    await set_user_game_prefs(uid, json.dumps({"grammar": False}))
    res = await build_session(uid, size=20)
    assert not [w for w in res["words"] if w.get("grammar")]
    assert res["composition"]["grammar"] == 0


async def test_overlay_skips_noun_without_forms(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "ting", "вещь")     # форм нет
    await _master(uid, pid)
    res = await build_session(uid, size=20)
    assert not [w for w in res["words"] if w.get("grammar")]


# ── (3) apply_result: грамм-клетка пишется, но base-рампа/«выучено» не страдают ─────────────
async def _modes(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT modes FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return json.loads(r["modes"]) if r and r["modes"] else {}


async def test_grammar_result_recorded_without_affecting_mastery(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)
    row = {"norwegian": "bok", "data": _DATA_NOUN}

    # верный род → клетка choice_gender зафиксирована
    await apply_result(uid, pid, True, mode="choice", direction="gender")
    m = await _modes(uid, pid)
    assert m.get("choice_gender") == "1"
    # все базовые клетки целы, слово по-прежнему выучено
    assert all(m.get(c) == "1" for c in REQUIRED_CELLS)
    assert _is_mastered(row, m)                       # «выучено» по base-рампе сохранилось

    # ОШИБКА в грамм-клетке НЕ откатывает base-рампу (в отличие от обычной клетки)
    await apply_result(uid, pid, False, mode="input", direction="indefpl")
    m2 = await _modes(uid, pid)
    assert m2.get("input_indefpl") == ""                  # грамм-клетка сброшена
    assert all(m2.get(c) == "1" for c in REQUIRED_CELLS)  # но базовые — нетронуты
    assert _is_mastered(row, m2)                          # «выучено» сохранилось


async def _srs_row(uid, pid):
    db = await _conn()
    try:
        async with db.execute(
            "SELECT strength, reps, lapses, ease, interval_days, due_at, correct, incorrect, streak "
            "FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return dict(r) if r else {}


async def test_grammar_answer_does_not_touch_base_srs(fresh_db):
    """Грамм-ответ (★) НЕ двигает SRS-расписание/счётчики выученного слова — ни ошибкой, ни верным.
    Раньше общий SRS-блок apply_result выполнялся и для грамм-клетки: ошибка в роде обнуляла interval/
    due/ease и плодила lapses (слово-«40 дней» сваливалось на повтор завтра). Теперь — ранний выход."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)
    before = await _srs_row(uid, pid)
    assert before.get("interval_days", 0) >= 1            # слово реально «выучено» с интервалом

    await apply_result(uid, pid, False, mode="choice", direction="gender")   # ОШИБКА в роде
    assert await _srs_row(uid, pid) == before, "грамм-ошибка изменила base-SRS выученного слова"

    await apply_result(uid, pid, True, mode="input", direction="indefpl")    # ВЕРНЫЙ ввод мн.ч.
    assert await _srs_row(uid, pid) == before, "верный грамм-ответ изменил base-SRS выученного слова"
