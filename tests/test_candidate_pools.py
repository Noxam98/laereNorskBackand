"""ЭТАП 4: чистые источники кандидатов (session/pools.py) — без БД, на голых данных."""
from session import pools


def _e(pid, no=None, pos="noun", status="review", due=False, lapses=0, strength=50,
       due_at="2026-01-01", last_seen=None, correct=1, incorrect=0):
    no = no or f"w{pid}"   # разные написания по умолчанию — иначе омограф-дедуп съест
    return {"row": {"pool_id": pid, "norwegian": no, "lapses": lapses, "strength": strength,
                    "due_at": due_at, "last_seen": last_seen, "correct": correct,
                    "incorrect": incorrect, "archived": None, "known": None},
            "modes": {}, "status": status, "due": due,
            "data": {"part_of_speech": pos}}


STEP = ("choice_int2no", "choice", "int2no")


def _select(pools_list):
    return pools.select_candidates(
        pools_list,
        next_step=lambda e: STEP, review_step=lambda e: STEP,
        func_locked=lambda e: False)


def test_pool_priority_order():
    """returned → overdue → due_mastered → weak → maturing."""
    returned = _e(1, lapses=2, status="learning")
    over = _e(2, due=True)
    dm = _e(3, status="mastered", due=True)
    wk = _e(4, status="weak")
    mat = _e(5, status="learning")
    built = pools.build_pools([mat, wk, dm, over, returned],
                              is_certified=lambda r: False,
                              is_function_word=lambda no, d: False)
    cand = _select(built)
    assert [c[0]["row"]["pool_id"] for c in cand] == [1, 2, 3, 4, 5]


def test_homograph_dedup_by_spelling():
    """Одно НАПИСАНИЕ на сессию: и (norwegian,pos)-дубли, и РАЗНО-pos омонимы схлопываются в один —
    оба смысла «liv» подряд в одной сессии сбивают юзера; второй смысл придёт в другой сессии."""
    a = _e(1, no="liv", pos="noun", due=True)
    b = _e(2, no="liv", pos="noun", due=True)     # дубль (norwegian, pos)
    c = _e(3, no="liv", pos="verb", due=True)     # другой смысл — тоже придерживаем
    built = pools.build_pools([a, b, c], is_certified=lambda r: False,
                              is_function_word=lambda no, d: False)
    cand = _select(built)
    assert [x[0]["row"]["pool_id"] for x in cand] == [1]


def test_cooldown_moves_to_tail_not_drops():
    fresh = (_e(1, last_seen="2026-07-04T10:00:00"), STEP)
    old = (_e(2, last_seen="2026-07-04T08:00:00"), STEP)
    out = pools.apply_cooldown([fresh, old], cooldown_cut="2026-07-04T09:00:00")
    assert [c[0]["row"]["pool_id"] for c in out] == [2, 1]   # свежий — в хвост, не выброшен


def test_mastered_only_via_review_pool():
    dm = _e(3, status="mastered", due=True)
    built = pools.build_pools([dm], is_certified=lambda r: False,
                              is_function_word=lambda no, d: False)
    # due_mastered идёт review-веткой; в обычных пулах mastered отсеян
    cand = _select(built)
    assert len(cand) == 1


def test_early_review_respects_core_contract():
    """Регрессионный замок инцидента #4: если review_step (ядро) вернул None —
    слово в досрочные повторы не попадает."""
    listen_word = _e(1, status="review", due_at="2026-01-02")
    normal = _e(2, status="review", due_at="2026-01-03")
    out = pools.early_review_pool(
        [listen_word, normal], size=10,
        review_step=lambda e: None if e["row"]["pool_id"] == 1 else STEP)
    assert [c[0]["row"]["pool_id"] for c in out] == [2]


def test_early_review_orders_by_nearest_due():
    a = _e(1, status="learning", due_at="2026-01-05")
    b = _e(2, status="learning", due_at="2026-01-02")
    out = pools.early_review_pool([a, b], size=10, review_step=lambda e: STEP)
    assert [c[0]["row"]["pool_id"] for c in out] == [2, 1]
