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
    _grammar_cells, _grammar_element, REQUIRED_CELLS, GRAMMAR_RATIO,
)
from db.learning_forms import apply_form_result
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
async def test_overlay_surfaces_for_mastered_noun(fresh_db, monkeypatch):
    """Цикл «слова↔формы»: выученное слово наполняет партию; партия готова → ФАЗА ФОРМ —
    сессия дрилит формы (карточка формы первой), новых слов не вводит."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)                                   # партия из 1 → фаза форм
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "forms"
    assert res["composition"]["fresh"] == 0                    # новых слов в фазе форм нет
    grams = [w for w in res["words"] if w.get("grammar")]
    assert grams, "фаза форм должна дрилить формы выученного сущ."
    g = grams[0]
    assert g["pool_id"] == pid and g["form_track"] is True
    assert g["step"] in ("gender", "indef_pl", "def_sg", "def_pl")
    assert g["stage"] == "card" and g["mode"] == "study"      # новая клетка → карточка формы
    assert res["composition"]["grammar"] == len(grams)
    assert len(grams) <= 2                                     # ≤2 клетки на слово за сессию


async def test_overlay_quota_proportional(fresh_db):
    """15 выученных → на 10-м партия готова (дефолт FORM_CYCLE_BATCH=10) → фаза форм.
    ПЕРВАЯ сессия партии: все клетки новые → только ПОРЦИЯ карточек (как «новых за сессию»,
    дефолт 6), стена из 14 карточек не валится; остальное сессия добьёт словами."""
    uid, did = await seed_user()
    for i in range(15):
        pid, _ = await seed_word(did, f"bok{i}", f"книга{i}")
        await _set_forms(pid, _BOK)
        await _master(uid, pid)
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "forms"
    assert res["composition"]["formsLeft"] == 10               # партия = 10, не 15
    assert res["composition"]["formsCellsLeft"] == 40          # 10 слов × 4 клетки — до новых слов
    grams = [w for w in res["words"] if w.get("grammar")]
    assert len(grams) == 6                                     # порция карточек (NEW_PER_SESSION)
    assert all(g["stage"] == "card" for g in grams)


async def test_forms_phase_card_portion_then_exercises(fresh_db):
    """Порция карточек: после просмотра карточек след. сессия несёт УПРАЖНЕНИЯ (choose) по ним
    + новую порцию карточек — формы разбавляются заданиями, а не листаются пачкой."""
    from db.learning_forms import FORMS_SESSION_SHARE
    uid, did = await seed_user()
    pids = []
    for i in range(12):
        pid, _ = await seed_word(did, f"bok{i}", f"книга{i}")
        await _set_forms(pid, _BOK)
        await _master(uid, pid)
        pids.append(pid)
    res = await build_session(uid, size=20)
    cards = [w for w in res["words"] if w.get("form_track")]
    assert len(cards) == 6 and all(w["stage"] == "card" for w in cards)
    for w in cards:                                            # карточки просмотрены
        await apply_form_result(uid, w["pool_id"], w["step"], True, stage="card")
    res2 = await build_session(uid, size=20)
    forms2 = [w for w in res2["words"] if w.get("form_track")]
    chooses = [w for w in forms2 if w["stage"] == "choose"]
    cards2 = [w for w in forms2 if w["stage"] == "card"]
    assert len(chooses) == 6                                   # вчерашние карточки стали выбором
    assert 1 <= len(cards2) <= 6                               # плюс новая порция карточек
    assert len(forms2) <= round(20 * FORMS_SESSION_SHARE)


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


# ── VERB-срез: ввод нерегулярных форм (членство в слабых классах, НЕ «≠ канону») ──────────────
_DATA_VERB = json.dumps({"part_of_speech": "verb", "translate": {"ru": ["x"]}})
_SKRIVE = {"pos": "verb", "present": "skriver", "past": "skrev", "perfect": "har skrevet"}   # сильный
_LAGE = {"pos": "verb", "present": "lager", "past": "laget", "perfect": "har laget"}         # слабый двухклассовый
_SNAKKE = {"pos": "verb", "present": "snakker", "past": "snakket", "perfect": "har snakket"}  # регуляр


def test_grammar_cells_verb_strong():
    # сильный: претерит skrev и причастие skrevet НЕ выводятся слабым правилом → дриллим оба
    assert _grammar_cells("skrive", _DATA_VERB, _SKRIVE) == ["input_past", "input_perfect"]


def test_grammar_cells_verb_weak_dualclass_empty():
    # слабый двухклассовый (lage→laget валидно) — НЕ дриллим: членство в слабых классах, не «≠ канону»
    assert _grammar_cells("lage", _DATA_VERB, _LAGE) == []
    assert _grammar_cells("snakke", _DATA_VERB, _SNAKKE) == []


# ── ADJECTIVE-срез: согласование/степени, когда факт ≠ правилу (или супплетив) ────────────────
_DATA_ADJ = json.dumps({"part_of_speech": "adjective", "translate": {"ru": ["x"]}})
_GOD = {"pos": "adjective", "neuter": "godt", "plural": "gode", "comparative": "bedre", "superlative": "best"}
_FIN = {"pos": "adjective", "neuter": "fint", "plural": "fine", "comparative": "finere", "superlative": "finest"}


def test_grammar_cells_adj_suppletive_degrees():
    # god→bedre→best: степени супплетивны → дриллим; ср.род/мн. регулярны → нет
    assert _grammar_cells("god", _DATA_ADJ, _GOD) == ["input_comparative", "input_superlative"]


def test_grammar_cells_adj_regular_empty():
    assert _grammar_cells("fin", _DATA_ADJ, _FIN) == []


def test_grammar_element_input_form_shape():
    # любая input-форма: mode input, direction = суффикс клетки, target из forms, строгая сверка; pos параметризован
    el = _grammar_element({"pool_id": 9, "norwegian": "skrive", "mastered": 1}, "input_past", _SKRIVE,
                          {"part_of_speech": "verb"})
    assert el["mode"] == "input" and el["direction"] == "past" and el["grammar"] is True
    assert el["part_of_speech"] == "verb"                 # pos НЕ захардкожен noun
    assert el["target"] == {"field": "past", "value": "skrev"}
    assert el["prompt"]["formLabel"] == "past" and el["scoring"] == {"typoForgive": False}
    ela = _grammar_element({"pool_id": 9, "norwegian": "god", "mastered": 1}, "input_superlative", _GOD,
                           {"part_of_speech": "adjective"})
    assert ela["direction"] == "superlative" and ela["target"]["value"] == "best"


async def test_overlay_surfaces_for_mastered_verb(fresh_db, monkeypatch):
    """Глагол — тоже трек форм: в фазе форм первая клетка (present/past/perfect) карточкой."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "skrive", "писать", pos="verb")
    await _set_forms(pid, _SKRIVE)
    await _master(uid, pid)
    res = await build_session(uid, size=20)
    grams = [w for w in res["words"] if w.get("grammar")]
    assert grams and grams[0]["pool_id"] == pid
    assert grams[0]["form_track"] is True and grams[0]["stage"] == "card"
    assert grams[0]["step"] in ("present", "past", "perfect")


async def test_words_phase_no_new_form_cells(fresh_db):
    """ФАЗА СЛОВ (партия не набрана, базе есть что отдать): клетки форм НЕ вводятся вовсе."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    await _set_forms(pid, _BOK)
    await _master(uid, pid)                                   # партия 1 из 10 — фаза остаётся words
    await seed_word(did, "hus", "дом")                        # живое НОВОЕ слово — база не голодна
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "words"
    assert not [w for w in res["words"] if w.get("form_track")]
    assert res["composition"]["total"] > 0                    # база отдала карточку нового


# ── PRONOUN/DETERMINER-срез: курируемая парадигма (форм НЕТ в БД, берём из PRONOUN_PARADIGM) ───
_DATA_PRON = json.dumps({"part_of_speech": "pronoun", "translate": {"ru": ["x"]}})
_DATA_DET = json.dumps({"part_of_speech": "determiner", "translate": {"ru": ["x"]}})


def test_grammar_cells_pronoun_objcase():
    # личное местоимение: объектный падеж (jeg→meg, han→ham), forms=None → курируемая парадигма
    assert _grammar_cells("jeg", _DATA_PRON, None) == ["input_objcase"]
    assert _grammar_cells("han", _DATA_PRON, None) == ["input_objcase"]


def test_grammar_cells_possessive_agreement():
    assert _grammar_cells("min", _DATA_DET, None) == ["input_possneut", "input_posspl"]


def test_grammar_cells_pronoun_not_in_paradigm_empty():
    assert _grammar_cells("det", _DATA_PRON, None) == []     # obj=det совпадает → дриллить нечего
    assert _grammar_cells("xyz", _DATA_PRON, None) == []     # не в парадигме


async def test_overlay_surfaces_for_mastered_pronoun(fresh_db, monkeypatch):
    """Местоим-overlay живёт в ФАЗЕ ФОРМ (в фазе слов грамматики нет вовсе)."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "jeg", "я", pos="pronoun")
    await _master(uid, pid)                       # форм в БД НЕ ставим — берётся из парадигмы
    res = await build_session(uid, size=20)       # фаза слов (местоимение партию не наполняет)
    assert not [w for w in res["words"] if w.get("grammar")], "фаза слов — без грамматики"
    pn, _ = await seed_word(did, "bok", "книга")  # формо-способное слово → партия → фаза форм
    await _set_forms(pn, _BOK)
    await _master(uid, pn)
    res2 = await build_session(uid, size=20)
    assert res2["composition"]["phase"] == "forms"
    ov = [w for w in res2["words"] if w.get("grammar") and not w.get("form_track")]
    assert ov and ov[0]["pool_id"] == pid and ov[0]["step"] == "input_objcase"
    assert ov[0]["target"]["value"] == "meg"


async def test_overlay_respects_per_pos_toggle(fresh_db, monkeypatch):
    """Пер-POS тумблер (gamePrefs.grammarPos): отключённая часть речи не даёт грамм-упражнений,
    включённая — даёт (в фазе форм). Здесь глаголы выключены, существительные — нет."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 2)
    uid, did = await seed_user()
    pn, _ = await seed_word(did, "bok", "книга", pos="noun")
    await _set_forms(pn, _BOK)
    await _master(uid, pn)
    pv, _ = await seed_word(did, "skrive", "писать", pos="verb")
    await _set_forms(pv, _SKRIVE)
    await _master(uid, pv)                                     # партия из 2 → фаза форм
    await set_user_game_prefs(uid, json.dumps({"grammarPos": {"verb": False}}))
    res = await build_session(uid, size=20)
    grams = {w["pool_id"] for w in res["words"] if w.get("grammar")}
    assert pn in grams and pv not in grams        # сущ. остаётся, глаг. отключён
