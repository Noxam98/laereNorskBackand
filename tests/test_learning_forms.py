"""Тесты трека форм: чистая логика (клетки, диспетч опций, элементы, планировщик) +
интеграция с БД (apply_form_result → form_srs → build_session)."""
import json
from db.core import _conn, _release
from db.learning import build_session, apply_result, REQUIRED_CELLS
from db.learning_forms import (
    FORM_STAGES, form_cells_for, form_options, form_element, cell_value, schedule_form,
    apply_form_result, load_form_states,
)
from tests.conftest import seed_user, seed_word


def _data(pos):
    return json.dumps({"part_of_speech": pos, "translate": {"ru": "x"}})


def test_form_cells_for():
    # сущ.: только присутствующие формы
    n = {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}
    assert set(form_cells_for("noun", n)) == {"gender", "indef_pl", "def_sg", "def_pl"}
    assert form_cells_for("noun", {"gender": "en"}) == ["gender"]        # пустые формы пропущены
    # глаг.: present/past/perfect
    v = {"present": "går", "past": "gikk", "perfect": "har gått"}
    assert set(form_cells_for("verb", v)) == {"present", "past", "perfect"}
    # прил.: перифраз 'mer praktisk' — не одна словоформа → мимо
    a = {"neuter": "praktisk", "plural": "praktiske", "comparative": "mer praktisk", "superlative": "mest praktisk"}
    assert set(form_cells_for("adjective", a)) == {"neuter", "plural"}
    # местоимение — морфологии форм нет
    assert form_cells_for("pronoun", {"obj": "meg"}) == []


def test_cell_value_perfect_participle():
    # перфект глагола дрилим как причастие (без 'har')
    assert cell_value("verb", {"perfect": "har gått"}, "perfect") == "gått"
    assert cell_value("noun", {"gender": "et"}, "gender") == "et"


def test_junk_form_markers_not_drilled():
    """LLM-маркеры «формы нет» (n/a и т.п.) — клетка не создаётся (не просим ввести «n/a»)."""
    f = {"neuter": "neste", "plural": "neste", "comparative": "n/a", "superlative": "N/A"}
    assert cell_value("adjective", f, "comparative") == ""
    assert set(form_cells_for("adjective", f)) == {"neuter", "plural"}
    assert cell_value("noun", {"gender": "-"}, "gender") == ""


def test_form_options_dispatch():
    # сущ. gender → 2 других артикля
    c, d = form_options("noun", "bil", {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}, "gender")
    assert c == "en" and set(d) == {"ei", "et"}
    # глаг. сильный: наивная слабая форма среди дистракторов
    c, d = form_options("verb", "gå", {"present": "går", "past": "gikk", "perfect": "har gått"}, "past")
    assert c == "gikk" and "gådde" in d
    # прил.: классическая ошибка согласования
    c, d = form_options("adjective", "grønn", {"neuter": "grønt", "plural": "grønne", "comparative": "grønnere", "superlative": "grønnest"}, "neuter")
    assert c == "grønt" and "grønnt" in d


def test_form_element_stages():
    row = {"pool_id": 7, "norwegian": "gå", "mastered": 1}
    forms = {"present": "går", "past": "gikk", "perfect": "har gått"}
    data = _data("verb")
    # card — пассивный показ формы
    card = form_element(row, forms, data, "past", "card")
    assert card["mode"] == "study" and card["reveal"] == "gikk" and card["form_track"] is True
    # choose — варианты содержат correct + дистракторы, correct первым в target
    ch = form_element(row, forms, data, "past", "choose")
    ws = [o["w"] for o in ch["options"]]
    assert ch["mode"] == "choice" and ch["target"]["value"] == "gikk" and "gikk" in ws and "gådde" in ws
    assert ch["target"]["value"] not in ch["distractors"]
    # produce — ввод формы
    pr = form_element(row, forms, data, "past", "produce")
    assert pr["mode"] == "input" and pr["target"]["value"] == "gikk"


def test_schedule_form_ramp():
    # card → choose, ещё в этой сессии
    assert schedule_form("card", 2.5, 0, True)[0] == "choose"
    assert schedule_form("card", 2.5, 0, True)[3] == 0
    # choose верно → produce, всё ещё в сессии (due 0)
    ns, ease, iv, due = schedule_form("choose", 2.5, 0, True)
    assert ns == "produce" and due == 0 and ease > 2.5
    # produce верно → клетка отработана, планируем повтор (due ≥ 1 день)
    ns, ease, iv, due = schedule_form("produce", 2.5, 0, True)
    assert ns == "produce" and iv >= 1 and due >= 1
    # produce ошибка → шаг назад к choose, ease вниз, повтор в сессии
    ns, ease, iv, due = schedule_form("produce", 2.5, 5, False)
    assert ns == "choose" and ease < 2.5 and due == 0
    # choose ошибка → назад к card
    assert schedule_form("choose", 2.5, 0, False)[0] == "card"
    # интервал растёт мультипликативно и упирается в потолок
    _, _, iv, _ = schedule_form("produce", 3.0, 200, True)
    assert iv <= 365


def test_form_stages_order():
    assert FORM_STAGES == ("card", "choose", "produce")


# ── Интеграция с БД: form_srs + сессия ────────────────────────────────────────

_GIKK = {"pos": "verb", "present": "går", "past": "gikk", "perfect": "har gått"}


async def _set_forms(pool_id, forms):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?", (json.dumps(forms), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def _master(uid, pid):
    for cell in REQUIRED_CELLS:
        mode, direction = cell.split("_", 1)
        await apply_result(uid, pid, True, mode=mode, direction=direction)


async def test_apply_form_result_ramp_progression(fresh_db):
    """card→choose→produce→интервальный повтор; ошибка откатывает ступень; base-SRS не тронут."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)

    # карточка показана → ступень choose, due сразу (повтор в след. сессии)
    r = await apply_form_result(uid, pid, "past", True, stage="card")
    assert r["stage"] == "choose"
    st = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st["stage"] == "choose" and st["reps"] == 0        # карточка — не ответ

    # выбор верно → produce (ещё в работе), ответ засчитан
    r = await apply_form_result(uid, pid, "past", True)
    assert r["stage"] == "produce"
    st = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st["reps"] == 1 and st["lapses"] == 0

    # ввод верно → клетка отработана, интервал ≥ 1 день
    r = await apply_form_result(uid, pid, "past", True)
    assert r["stage"] == "produce"
    st = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st["interval_days"] >= 1 and st["due_at"] > st["last_seen"]

    # ошибка на повторе → откат к choose, lapse, скорый повтор
    r = await apply_form_result(uid, pid, "past", False)
    assert r["stage"] == "choose"
    st = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st["lapses"] == 1 and st["ease"] < 2.7


async def test_form_answers_do_not_touch_base_srs(fresh_db):
    """Ответы трека форм не двигают base-SRS слова (отдельный слой, как overlay)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    db = await _conn()
    try:
        async with db.execute("SELECT reps, ease, interval_days, due_at, modes FROM user_words "
                              "WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            before = dict(await cur.fetchone())
    finally:
        await _release(db)
    await apply_form_result(uid, pid, "past", True, stage="card")
    await apply_form_result(uid, pid, "past", False)
    await apply_form_result(uid, pid, "perfect", True)
    db = await _conn()
    try:
        async with db.execute("SELECT reps, ease, interval_days, due_at, modes FROM user_words "
                              "WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            after = dict(await cur.fetchone())
    finally:
        await _release(db)
    assert after == before


async def test_session_serves_stage_after_card(fresh_db, monkeypatch):
    """После карточки клетка снова в сессии УЖЕ упражнением-выбором (stage choose, options)."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)          # партия из одного — сразу фаза форм
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)

    res = await build_session(uid, size=20)
    el = next(w for w in res["words"] if w.get("form_track"))
    assert el["stage"] == "card"                            # первая встреча — карточка формы
    first_cell = el["step"]

    await apply_form_result(uid, pid, first_cell, True, stage="card")   # карточку посмотрели
    res2 = await build_session(uid, size=20)
    el2 = next(w for w in res2["words"] if w.get("form_track"))
    assert el2["step"] == first_cell and el2["stage"] == "choose"       # та же клетка, теперь выбор
    ws = [o["w"] for o in el2["options"]]
    assert el2["target"]["value"] in ws and len(ws) >= 2                # верный + дистракторы на месте
    assert el2["target"]["value"] not in el2["distractors"]


async def test_forms_phase_session_starts_with_forms(fresh_db, monkeypatch):
    """Сессия фазы форм НАЧИНАЕТСЯ с форм: base-повторы слов — в хвосте, не первыми
    (баг: формы эмитились после base-цикла и юзер открывал «сессию форм» с переводов)."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    # ещё слово в прогрессе — оно даст base-упражнение в той же сессии
    pid2, _ = await seed_word(did, "hus", "дом")
    await apply_result(uid, pid2, True, mode="choice", direction="int2no")
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "forms"
    kinds = [bool(w.get("form_track")) for w in res["words"]]
    assert kinds[0] is True                                   # первая — форма
    assert kinds == sorted(kinds, reverse=True)               # формы монолитом в начале


async def test_cycle_full_circle(fresh_db, monkeypatch):
    """Полный круг: выучил партию → фаза форм → сдал все клетки → снова фаза слов."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    assert (await lf.get_form_cycle(uid))["phase"] == "forms"      # партия готова

    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "forms" and res["composition"]["formsLeft"] == 1

    # сдаём все клетки: card → choose(верно) → produce(верно) — клетка graduates (interval ≥ 1)
    for cell in ("present", "past", "perfect"):
        await apply_form_result(uid, pid, cell, True, stage="card")
        await apply_form_result(uid, pid, cell, True)
        await apply_form_result(uid, pid, cell, True)

    res2 = await build_session(uid, size=20)                        # партия сдана → флип
    assert res2["composition"]["phase"] == "words"
    assert not [w for w in res2["words"] if w.get("form_track")]    # повторы ещё не подошли
    assert (await lf.get_form_cycle(uid)) == {"phase": "words", "batch": []}


async def test_words_phase_serves_due_form_reviews(fresh_db, monkeypatch):
    """Повторы форм (due) не замораживаются фазой слов: подошедшая клетка приходит в сессию."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 1)
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    for cell in ("present", "past", "perfect"):                     # сдаём партию → фаза words
        await apply_form_result(uid, pid, cell, True, stage="card")
        await apply_form_result(uid, pid, cell, True)
        await apply_form_result(uid, pid, cell, True)
    # состариваем один повтор — будто интервал прошёл
    db = await _conn()
    try:
        await db.execute("UPDATE form_srs SET due_at = '2000-01-01T00:00:00' "
                         "WHERE user_id=? AND pool_id=? AND cell='past'", (uid, pid))
        await db.commit()
    finally:
        await _release(db)
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "words"
    forms = [w for w in res["words"] if w.get("form_track")]
    assert len(forms) == 1 and forms[0]["step"] == "past"           # только due-повтор, без новых клеток
    assert forms[0]["stage"] == "produce"                           # сданная клетка повторяется вводом


async def test_cycle_veteran_seed(fresh_db, monkeypatch):
    """Ветеран без строки цикла (выучил слова ДО релиза): первый build сидит партию из бэклога."""
    import db.learning_forms as lf
    monkeypatch.setattr(lf, "FORM_CYCLE_BATCH", 2)
    uid, did = await seed_user()
    pids = []
    for i, w in enumerate(("gå", "se")):
        pid, _ = await seed_word(did, w, f"слово{i}", pos="verb")
        await _set_forms(pid, {"pos": "verb", "present": w + "r", "past": "x" + w, "perfect": "har y" + w})
        await _master(uid, pid)
        pids.append(pid)
    db = await _conn()                                              # имитация «до релиза»: строки цикла нет
    try:
        await db.execute("DELETE FROM form_cycle WHERE user_id = ?", (uid,))
        await db.commit()
    finally:
        await _release(db)
    res = await build_session(uid, size=20)
    assert res["composition"]["phase"] == "forms"                   # сид из бэклога → сразу формы
    cyc = await lf.get_form_cycle(uid)
    assert cyc["phase"] == "forms" and len(cyc["batch"]) == 2 and set(cyc["batch"]) == set(pids)


def test_gender_produce_is_choice():
    """Produce-ступень рода — тоже ВЫБОР артикля (текстом артикль не набирают)."""
    row = {"pool_id": 3, "norwegian": "bok", "mastered": 1}
    forms = {"gender": "ei", "indef_pl": "bøker", "def_sg": "boka", "def_pl": "bøkene"}
    el = form_element(row, forms, {"part_of_speech": "noun"}, "gender", "produce")
    assert el["mode"] == "choice" and el["target"]["value"] == "ei"
    assert {o["w"] for o in el["options"]} == {"en", "ei", "et"}
    # обычная клетка на produce остаётся вводом
    el2 = form_element(row, forms, {"part_of_speech": "noun"}, "def_sg", "produce")
    assert el2["mode"] == "input"


def test_gender_feminine_accepts_common():
    """ei-слово валидно и как общего рода (реформа 2005): выбор «en» не наказываем,
    карточка учит «ei/en»; мужское слово (en) — без accept (ei bil не бывает)."""
    row = {"pool_id": 3, "norwegian": "bok", "mastered": 1}
    fei = {"gender": "ei", "indef_pl": "bøker", "def_sg": "boka", "def_pl": "bøkene"}
    el = form_element(row, fei, {"part_of_speech": "noun"}, "gender", "choose")
    assert el["target"]["value"] == "ei" and el["target"]["accept"] == ["en"]
    card = form_element(row, fei, {"part_of_speech": "noun"}, "gender", "card")
    assert card["reveal"] == "ei/en bok"
    fen = {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}
    el2 = form_element({"pool_id": 4, "norwegian": "bil", "mastered": 1}, fen,
                       {"part_of_speech": "noun"}, "gender", "choose")
    assert el2["target"]["value"] == "en" and "accept" not in el2["target"]
    card2 = form_element({"pool_id": 4, "norwegian": "bil", "mastered": 1}, fen,
                         {"part_of_speech": "noun"}, "gender", "card")
    assert card2["reveal"] == "en bil"


async def test_stats_forms_block(fresh_db):
    """learning_stats отдаёт блок forms: клеток в работе / отработано / к повторению."""
    from db.learning_suggest import learning_stats
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    s0 = await learning_stats(uid)
    assert s0["forms"] == {"cells": 0, "done": 0, "due": 0}
    await apply_form_result(uid, pid, "past", True, stage="card")     # card→choose, due сразу
    await apply_form_result(uid, pid, "perfect", True, stage="card")
    await apply_form_result(uid, pid, "past", True)                   # choose→produce (due сразу)
    await apply_form_result(uid, pid, "past", True)                   # produce сдан → interval≥1
    s1 = await learning_stats(uid)
    assert s1["forms"]["cells"] == 2
    assert s1["forms"]["done"] == 1        # past отработана
    assert s1["forms"]["due"] == 1         # perfect ждёт выбора сейчас


async def test_stale_bundle_form_answer_does_not_pollute_base(fresh_db):
    """Старый бандл шлёт ответ формы БЕЗ form-флага (mode=choice, direction=past) →
    base-SRS выученного слова не двигается (защитный гард apply_result)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    db = await _conn()
    try:
        async with db.execute("SELECT reps, ease, interval_days, due_at FROM user_words "
                              "WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            before = dict(await cur.fetchone())
    finally:
        await _release(db)
    await apply_result(uid, pid, False, mode="choice", direction="past")    # ошибка «выбора формы»
    await apply_result(uid, pid, True, mode="choice", direction="def_sg")   # верный «выбор формы»
    db = await _conn()
    try:
        async with db.execute("SELECT reps, ease, interval_days, due_at FROM user_words "
                              "WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            after = dict(await cur.fetchone())
    finally:
        await _release(db)
    assert after == before
