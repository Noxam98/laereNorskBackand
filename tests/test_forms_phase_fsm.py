"""ЭТАП 5: FSM цикла «слова↔формы» (session/forms_phase.py) — чисто, без БД.

Замки на реальные инциденты: сид ветерана, флип «партия сдана», анти-дедлок
words→forms, «грамматика выкл → cap_new НЕ обнулён», паритет is_formable
между build_session и apply_result.
"""
import json

from session import forms_phase as fp

NOW = "2026-07-04T12:00:00"
CELLS = ("c1", "c2", "c3")


def _e(pid, no=None, pos="noun", status="mastered", freq=0, forms=None, modes=None):
    return {"row": {"pool_id": pid, "norwegian": no or f"w{pid}", "freq": freq,
                    "forms": json.dumps(forms) if isinstance(forms, dict) else forms},
            "modes": modes or {}, "status": status, "due": False,
            "data": {"part_of_speech": pos}}


_GROUPS = {"noun": "noun", "verb": "verb", "adjective": "adjective", "pronoun": "pronoun"}


def _universe(enriched, *, ordered_pids=frozenset(), toggles=None):
    toggles = toggles or {}

    def group_of(e):
        g = _GROUPS.get(e["data"]["part_of_speech"])
        return g if g and toggles.get(g, True) else None

    return fp.build_universe(
        enriched, ordered_pids=ordered_pids, group_of=group_of,
        pronoun_forms={"min": {"m": "min", "f": "mi", "n": "mitt"}}.get,
        cells_for=lambda e, fdict: [c for c in CELLS if fdict.get(c)])


def _plan(cands, info, *, fstates=None, cycle_state=None, cap_new=4, size=10,
          base_servable=5, batch_size=3):
    return fp.plan_forms_phase(
        cands, info, fstates=fstates or {}, cycle_state=cycle_state, now_s=NOW,
        cap_new=cap_new, size=size, base_servable=base_servable,
        batch_size=batch_size, session_share=0.7,
        overlay_pending=lambda e, fdict: [c for c in ("ov1", "ov2")
                                          if e["modes"].get(c) != "1"])


FORMS = {"c1": "en bok", "c2": "boka", "c3": "bøker"}


# ── build_universe ───────────────────────────────────────────────────────────
def test_universe_excludes_ordered_and_unmastered_and_toggled_off():
    words = [_e(1, forms=FORMS), _e(2, forms=FORMS, status="review"),
             _e(3, forms=FORMS), _e(4, pos="verb", forms=FORMS)]
    cands, info = _universe(words, ordered_pids={3}, toggles={"verb": False})
    assert set(info) == {1}          # 2 не выучено, 3 в base-сессии, 4 выключен тумблером
    assert [e["row"]["pool_id"] for e, _, _ in cands] == [1]


def test_universe_pronoun_via_paradigm_not_in_track():
    words = [_e(1, no="min", pos="pronoun")]
    cands, info = _universe(words)
    assert [g for _, g, _ in cands] == ["pronoun"]
    assert info == {}                # местоимение — overlay, НЕ трек форм


# ── сид цикла (первый заход после релиза) ────────────────────────────────────
def test_seed_veteran_backlog_goes_straight_to_forms():
    words = [_e(p, freq=100 - p, forms=FORMS) for p in (1, 2, 3, 4)]
    cands, info = _universe(words)
    plan = _plan(cands, info, cycle_state=None, batch_size=3)
    assert plan["phase"] == "forms"
    assert plan["save_cycle"] == ("forms", [1, 2, 3])   # топ-частотные, партия = batch_size
    assert plan["cap_new"] == 0                         # фаза форм новых слов не вводит
    assert plan["picks"]


def test_seed_newbie_backlog_below_batch_stays_words():
    words = [_e(1, forms=FORMS)]
    cands, info = _universe(words)
    plan = _plan(cands, info, cycle_state=None, batch_size=3, cap_new=4)
    assert plan["phase"] == "words"
    assert plan["save_cycle"] == ("words", [])
    assert plan["picks"] == [] and plan["cap_new"] == 4


# ── флипы ────────────────────────────────────────────────────────────────────
def test_flip_forms_to_words_when_batch_done():
    """Партия сдана (все клетки produce+interval≥1) → по кругу: снова слова."""
    words = [_e(1, forms=FORMS)]
    cands, info = _universe(words)
    fstates = {(1, c): {"interval_days": 3, "due_at": "2099-01-01", "stage": "produce"}
               for c in CELLS}
    plan = _plan(cands, info, fstates=fstates,
                 cycle_state={"phase": "forms", "batch": [1]}, cap_new=4)
    assert plan["phase"] == "words"
    assert plan["save_cycle"] == ("words", [])
    assert plan["cap_new"] == 4      # вернулись к словам — квота новых живёт


def test_antideadlock_flip_words_to_forms():
    """Базе нечего отдать, а несданные клетки есть → флип в forms (пустых сессий нет)."""
    words = [_e(1, freq=10, forms=FORMS), _e(2, freq=5, forms=FORMS)]
    cands, info = _universe(words)
    plan = _plan(cands, info, cycle_state={"phase": "words", "batch": []},
                 base_servable=0, batch_size=10)
    assert plan["phase"] == "forms"
    assert plan["save_cycle"] == ("forms", [1, 2])
    # а если базе ЕСТЬ что отдать — остаёмся в words и цикл НЕ трогаем
    plan2 = _plan(cands, info, cycle_state={"phase": "words", "batch": []},
                  base_servable=3)
    assert plan2["phase"] == "words" and plan2["save_cycle"] is None


def test_no_flip_when_no_form_work():
    """База пуста, но и форм-работы нет → words, дёргать цикл незачем."""
    cands, info = _universe([])
    plan = _plan(cands, info, cycle_state={"phase": "words", "batch": []},
                 base_servable=0)
    assert plan["phase"] == "words" and plan["save_cycle"] is None


# ── отбор заданий фазы форм ──────────────────────────────────────────────────
def test_due_reviews_come_first_with_stored_stage():
    words = [_e(1, forms=FORMS)]
    cands, info = _universe(words)
    fstates = {(1, "c1"): {"interval_days": 2, "due_at": "2026-07-01", "stage": "choose"}}
    plan = _plan(cands, info, fstates=fstates,
                 cycle_state={"phase": "forms", "batch": [1]})
    kind, e, cell, _f, stage = plan["picks"][0]
    assert (kind, cell, stage) == ("form", "c1", "choose")   # повтор сданной клетки — первым


def test_cards_portioned_by_cap_and_two_cells_per_word():
    """Нетронутые клетки идут карточками ПОРЦИЯМИ (cards_cap=cap_new), ≤2 клетки на слово."""
    words = [_e(1, freq=10, forms=FORMS), _e(2, freq=5, forms=FORMS)]
    cands, info = _universe(words)
    # у слова 1 все три клетки НАЧАТЫ (interval<1 → не карточки, кап их не режет)
    started = {(1, c): {"interval_days": 0, "due_at": NOW, "stage": "choose"} for c in CELLS}
    plan = _plan(cands, info, fstates=started,
                 cycle_state={"phase": "forms", "batch": [1, 2]}, cap_new=1, size=20)
    cards = [p for p in plan["picks"] if p[4] == "card"]
    assert len(cards) == 1 and plan["cards_cap"] == 1       # порция карточек слова 2 = 1
    per_word = {}
    for _k, e, _c, _f, _s in plan["picks"]:
        per_word[e["row"]["pool_id"]] = per_word.get(e["row"]["pool_id"], 0) + 1
    assert per_word[1] == 2      # раунд-робин: ≤2 клетки на слово, хотя несданных у него 3


def test_overlay_fills_quota_remainder():
    words = [_e(1, no="min", pos="pronoun", freq=50)]
    cands, info = _universe(words)
    plan = _plan(cands, info, cycle_state={"phase": "words", "batch": []},
                 base_servable=0)
    # у местоимения форм-трека нет → сам по себе флип не случится и overlay не выдаётся
    assert plan["phase"] == "words" and plan["picks"] == []
    # но в фазе форм (партия другого слова) overlay добивает квоту
    words2 = words + [_e(2, forms=FORMS)]
    cands2, info2 = _universe(words2)
    plan2 = _plan(cands2, info2, cycle_state={"phase": "forms", "batch": [2]})
    assert any(k == "overlay" and e["row"]["norwegian"] == "min"
               for k, e, _c, _f, _s in plan2["picks"])


def test_cycle_counters_for_start_button():
    words = [_e(1, forms=FORMS), _e(2, forms={"c1": "x"})]
    cands, info = _universe(words)
    plan = _plan(cands, info, cycle_state={"phase": "forms", "batch": [1, 2]}, size=30)
    assert plan["cycle_left"] == 2          # слов партии с несданными формами
    assert plan["cycle_cells"] == 4         # 3 клетки + 1 клетка


# ── грамматика выключена / дрилл по набору ───────────────────────────────────
def test_disabled_plan_keeps_cap_new_intact():
    """Инцидент «отключил грамматику — не грузится сессия»: план пуст, но cap_new ЖИВ."""
    plan = fp.empty_plan(6)
    assert plan["picks"] == [] and plan["phase"] == "words"
    assert plan["cap_new"] == 6 and plan["save_cycle"] is None


# ── is_formable: единый предикат обоих краёв цикла ───────────────────────────
def test_is_formable_parity_with_form_cells_for():
    from db.learning_forms import is_formable, form_cells_for, parse_forms
    noun = json.dumps({"gender": "en", "indef_pl": "bøker", "def_pl": "bøkene"})
    for pos, forms in [("noun", noun), ("verb", "{}"), ("adjective", None),
                       ("noun", json.dumps({"gender": "n/a"}))]:
        expect = pos in ("noun", "verb", "adjective") and bool(
            form_cells_for(pos, parse_forms(forms), "x"))
        assert is_formable(pos, forms, "x") is expect
    assert is_formable("pronoun", noun, "min") is False   # местоимения — не трек форм
