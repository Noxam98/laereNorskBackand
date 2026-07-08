"""Багфиксы трека ФОРМ (session/forms_phase.py + db/learning_forms.py).

Замки на конкретные инциденты:
#1  due-повторы форм не съедают всю квоту → партия двигается, флип forms→words наступает,
    cap_new освобождается (ветеран с ≥quota повторов больше не застревает в форм-фазе);
#2  слово партии, временно взятое base как due-повтор (withheld), не схлопывает партию;
#3  apply_form_result сериализован пер-юзер локом (нет lost update при дабл-тапе/двух вкладках);
#4  produce-клетка несёт target.accept с валидными дублетами (boka/boken, kastet/kasta, husene/husa);
#5  клетка рода только для en/ei/et; феминное-как-'en' не отвергает валидный выбор;
#6  перфект 'har' без причастия не дриллится; uncountable как 1/'true' убирает мн.ч.
"""
import json
import asyncio

from session import forms_phase as fp
from tests.conftest import seed_user, seed_word

NOW = "2026-07-08T12:00:00"
PAST = "2000-01-01T00:00:00"
FUTURE = "2099-01-01T00:00:00"
CELLS = ("c1", "c2", "c3")
FORMS = {"c1": "en bok", "c2": "boka", "c3": "bøker"}


def _e(pid, freq=0, forms=FORMS, pos="noun", status="mastered"):
    return {"row": {"pool_id": pid, "norwegian": f"w{pid}", "freq": freq,
                    "forms": json.dumps(forms)},
            "modes": {}, "status": status, "due": False,
            "data": {"part_of_speech": pos}}


def _universe(words, ordered_pids=frozenset()):
    return fp.build_universe(
        words, ordered_pids=ordered_pids,
        group_of=lambda e: e["data"]["part_of_speech"],
        pronoun_forms={}.get,
        cells_for=lambda e, fdict: [c for c in CELLS if fdict.get(c)])


def _plan(cands, info, *, fstates=None, cycle_state=None, cap_new=4, size=10,
          base_servable=5, batch_size=3, withheld=frozenset()):
    return fp.plan_forms_phase(
        cands, info, fstates=fstates or {}, cycle_state=cycle_state, now_s=NOW,
        cap_new=cap_new, size=size, base_servable=base_servable,
        batch_size=batch_size, session_share=0.7, withheld=withheld,
        overlay_pending=lambda e, fdict: [])


# ── #1: due-повторы не душат партию ──────────────────────────────────────────
def test_due_reviews_do_not_starve_batch():
    """Ветеран: 33 подошедших повтора форм (>quota) + партия из 3 слов со свежими клетками.
    Резерв под прогресс не даёт повторам занять всю квоту — партия двигается."""
    batch_words = [_e(p, freq=100) for p in (1, 2, 3)]
    review_words = [_e(p, freq=1) for p in range(10, 21)]         # 11 слов × 3 клетки = 33 повтора
    cands, info = _universe(batch_words + review_words)
    fstates = {(p, c): {"interval_days": 5, "due_at": PAST, "stage": "produce"}
               for p in range(10, 21) for c in CELLS}             # review-клетки сданы и подошли
    plan = _plan(cands, info, fstates=fstates,
                 cycle_state={"phase": "forms", "batch": [1, 2, 3]}, size=10, cap_new=4)
    quota = max(1, round(10 * 0.7))                               # 7
    assert plan["phase"] == "forms" and plan["cap_new"] == 0
    assert len(plan["picks"]) == quota
    batch_picks = [p for p in plan["picks"] if p[1]["row"]["pool_id"] in (1, 2, 3)]
    review_picks = [p for p in plan["picks"] if p[1]["row"]["pool_id"] >= 10]
    assert batch_picks, "партия должна двигаться, а не стоять за стеной повторов"
    assert len(review_picks) <= quota - max(1, quota // 3)        # повторы не съели весь слот


def test_batch_flips_and_frees_new_words_when_complete():
    """Все клетки партии сданы → флип forms→words и cap_new возвращается (новые слова опять идут)."""
    words = [_e(p, freq=100) for p in (1, 2, 3)]
    cands, info = _universe(words)
    fstates = {(p, c): {"interval_days": 5, "due_at": FUTURE, "stage": "produce"}
               for p in (1, 2, 3) for c in CELLS}
    plan = _plan(cands, info, fstates=fstates,
                 cycle_state={"phase": "forms", "batch": [1, 2, 3]}, cap_new=5)
    assert plan["phase"] == "words" and plan["cap_new"] == 5
    assert plan["save_cycle"] == ("words", [])


# ── #2: withheld удерживает партию открытой ──────────────────────────────────
def test_withheld_review_word_keeps_batch_open():
    """Слово 3 партии в этом проходе взято base как due-повтор → выпало из info.
    Без withheld партия схлопывается (баг), с withheld={3} — сохраняется."""
    words = [_e(1, freq=30), _e(2, freq=20)]                      # слова 1,2 в info; 3 — нет
    cands, info = _universe(words)
    fstates = {(p, c): {"interval_days": 5, "due_at": FUTURE, "stage": "produce"}
               for p in (1, 2) for c in CELLS}                    # 1,2 сданы
    plan_old = _plan(cands, info, fstates=fstates,
                     cycle_state={"phase": "forms", "batch": [1, 2, 3]})
    assert plan_old["save_cycle"] == ("words", [])               # преждевременный флип+очистка (баг)
    plan_new = _plan(cands, info, fstates=fstates,
                     cycle_state={"phase": "forms", "batch": [1, 2, 3]}, withheld={3})
    assert plan_new["save_cycle"] is None                        # партию НЕ очистили
    assert plan_new["phase"] == "words" and plan_new["cap_new"] == 4   # слова этой сессии, cap жив


def test_withheld_reviews_only_ordered_mastered_formable():
    """withheld_reviews: только выученные формо-способные слова трека, выпавшие из info из-за base."""
    words = [_e(1), _e(2, status="review"), _e(3), _e(4, pos="pronoun")]
    got = fp.withheld_reviews(
        words, ordered_pids={1, 2, 3, 4},
        group_of=lambda e: e["data"]["part_of_speech"],
        cells_for=lambda e, fdict: [c for c in CELLS if fdict.get(c)])
    assert got == {1, 3}          # 2 не выучено, 4 — местоимение (не трек форм)


# ── #3: пер-юзер лок на apply_form_result ────────────────────────────────────
_GIKK = {"pos": "verb", "present": "går", "past": "gikk", "perfect": "har gått"}


async def _set_forms(pool_id, forms):
    from db.core import _conn, _release
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?", (json.dumps(forms), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def _master(uid, pid):
    from db.learning import apply_result, REQUIRED_CELLS
    for cell in REQUIRED_CELLS:
        mode, direction = cell.split("_", 1)
        await apply_result(uid, pid, True, mode=mode, direction=direction)


async def test_apply_form_result_serialized_under_lock(fresh_db):
    """Два одновременных ответа на одну клетку: лок сериализует read-modify-write (нет lost update)."""
    from db.learning_forms import apply_form_result, load_form_states
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "gå", "идти", pos="verb")
    await _set_forms(pid, _GIKK)
    await _master(uid, pid)
    await apply_form_result(uid, pid, "past", True, stage="card")   # card→choose
    await apply_form_result(uid, pid, "past", True)                 # choose→produce, reps=1
    st0 = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st0["stage"] == "produce" and st0["reps"] == 1
    # оба ответа должны быть учтены (1→2→3), а не затёрты вторым коммитом (было бы 2)
    await asyncio.gather(apply_form_result(uid, pid, "past", True),
                         apply_form_result(uid, pid, "past", True))
    st = (await load_form_states(uid, [pid]))[(pid, "past")]
    assert st["reps"] == 3


# ── #4: produce несёт accept с дублетами ─────────────────────────────────────
def test_produce_carries_accept_doublets():
    from db.learning_forms import form_element
    row = {"pool_id": 1, "norwegian": "bok", "mastered": 1}
    fbok = {"gender": "ei", "indef_pl": "bøker", "def_sg": "boka", "def_pl": "bøkene"}
    el = form_element(row, fbok, {"part_of_speech": "noun"}, "def_sg", "produce")
    assert el["mode"] == "input" and el["target"]["value"] == "boka"
    assert "boken" in el["target"]["accept"]                       # феминное опр.ед.: boka/boken

    rowh = {"pool_id": 2, "norwegian": "hus", "mastered": 1}
    fhus = {"gender": "et", "indef_pl": "hus", "def_sg": "huset", "def_pl": "husene"}
    elh = form_element(rowh, fhus, {"part_of_speech": "noun"}, "def_pl", "produce")
    assert "husa" in elh["target"]["accept"]                       # опр.мн. среднего: husene/husa

    rowk = {"pool_id": 3, "norwegian": "kaste", "mastered": 1}
    fk = {"present": "kaster", "past": "kastet", "perfect": "har kastet"}
    elk = form_element(rowk, fk, {"part_of_speech": "verb"}, "past", "produce")
    assert "kasta" in elk["target"]["accept"]                      # слабый претерит: kastet/kasta

    rowg = {"pool_id": 4, "norwegian": "gå", "mastered": 1}
    fg = {"present": "går", "past": "gikk", "perfect": "har gått"}
    elg = form_element(rowg, fg, {"part_of_speech": "verb"}, "past", "produce")
    assert "accept" not in elg["target"]                          # сильный глагол — дублетов нет


# ── #5: клетка рода — только en/ei/et; феминное-как-en не отвергает верное ────
def test_gender_cell_gated_and_feminine_en_no_false_ei():
    from db.learning_forms import cell_value, form_cells_for, form_element
    assert cell_value("noun", {"gender": "m"}, "gender") == ""
    assert cell_value("noun", {"gender": "hankjønn"}, "gender") == ""
    assert cell_value("noun", {"gender": "en"}, "gender") == "en"
    assert "gender" not in form_cells_for("noun", {"gender": "m", "def_sg": "bilen"}, "bil")
    assert "gender" in form_cells_for("noun", {"gender": "en", "def_sg": "bilen"}, "bil")
    # феминное, записанное как 'en': 'ei' не даём ложным дистрактором (иначе отвергаем верный выбор)
    row = {"pool_id": 5, "norwegian": "bil", "mastered": 1}
    fen = {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}
    el = form_element(row, fen, {"part_of_speech": "noun"}, "gender", "choose")
    ws = {o["w"] for o in el["options"]}
    assert "ei" not in ws and "en" in ws and "et" in ws


# ── #6: перфект 'har' без причастия + мягкая истинность uncountable ───────────
def test_perfect_bare_aux_not_drilled():
    from db.learning_forms import cell_value, form_cells_for
    assert cell_value("verb", {"perfect": "har"}, "perfect") == ""
    assert cell_value("verb", {"perfect": "har "}, "perfect") == ""
    assert cell_value("verb", {"perfect": "er"}, "perfect") == ""
    cells = form_cells_for("verb", {"present": "går", "past": "gikk", "perfect": "har"})
    assert "perfect" not in cells and "past" in cells


def test_uncountable_truthy_coercions_skip_plural():
    from db.learning_forms import form_cells_for
    base = {"gender": "en", "indef_pl": "bruker", "def_sg": "bruken", "def_pl": "brukene"}
    for flag in (True, 1, "true", "1", "yes"):
        cells = set(form_cells_for("noun", {**base, "uncountable": flag}, "melodi"))
        assert cells == {"gender", "def_sg"}, f"flag={flag!r} должен убрать мн.ч."
    for flag in (False, 0, "false", None, ""):
        cells = set(form_cells_for("noun", {**base, "uncountable": flag}, "melodi"))
        assert "indef_pl" in cells and "def_pl" in cells, f"flag={flag!r} — мн.ч. остаётся"
