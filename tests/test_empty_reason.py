"""ЭТАП 8: диагностика comp.reason — единая лестница session/reason.py.

Табличный приоритет early_review → listen_pending → exam_pending → wip_full →
all_done. Раньше «early_review» ставился inline отдельно от empty-веток
(srs.gates.empty_session_reason) — два дома критериев; теперь один.
"""
import pytest

from session.reason import session_reason

WIP = 20


def _r(**over):
    kw = dict(session_len=0, scoped=False, early_review=False, has_audio_pending=False,
              gate_open=False, in_work=0, wip_limit=WIP)
    kw.update(over)
    return session_reason(**kw)


def test_nonempty_normal_session_has_no_reason():
    assert _r(session_len=4) is None


def test_early_review_only_when_session_nonempty():
    assert _r(session_len=3, early_review=True) == "early_review"
    # early_review-флаг, но сессия пуста → это НЕ early_review (падаем ниже по лестнице)
    assert _r(session_len=0, early_review=True, in_work=5) == "all_done"


@pytest.mark.parametrize("kw,expected", [
    (dict(has_audio_pending=True, gate_open=True, in_work=99), "listen_pending"),
    (dict(has_audio_pending=False, gate_open=True, in_work=99), "exam_pending"),
    (dict(has_audio_pending=False, gate_open=False, in_work=WIP), "wip_full"),
    (dict(has_audio_pending=False, gate_open=False, in_work=0), "all_done"),
])
def test_empty_session_priority(kw, expected):
    """Слух → экзамен → перебор → всё повторено; порядок фиксирован."""
    assert _r(session_len=0, **kw) == expected


def test_early_review_wins_over_empty_ladder():
    """Непустая досрочная сессия важнее любой empty-причины (её и не должно быть)."""
    assert _r(session_len=2, early_review=True, has_audio_pending=True,
              gate_open=True, in_work=99) == "early_review"


def test_scoped_never_gets_reason():
    """Дрилл по набору — вне диагностики дневного конвейера."""
    assert _r(session_len=0, scoped=True, gate_open=True) is None
    assert _r(session_len=3, scoped=True, early_review=True) is None
