"""ЭТАП 3: бюджеты/ворота — юнит-матрица именованных правил srs/gates.py."""
import pytest

from srs import gates
from srs.cells import CONTENT, PHRASE, REQUIRED_CELLS, AUDIO_CELL

AUDIO_PENDING = {c: "1" for c in REQUIRED_CELLS if c != AUDIO_CELL}


# ── counts_toward_wip: инцидент #3 как юнит ──────────────────────────────────
@pytest.mark.parametrize("status,expected", [
    ("learning", True), ("review", True), ("weak", True),
    ("new", False), ("mastered", False), ("archived", False), ("known", False),
])
def test_wip_by_status(status, expected):
    assert gates.counts_toward_wip(status, CONTENT, {}, audio_on=False) is expected


def test_wip_audio_pending_word_takes_no_slot():
    """Слово, ждущее ТОЛЬКО слуха, не занимает слот «в работе» (тупик 3.07)."""
    assert gates.counts_toward_wip("review", CONTENT, AUDIO_PENDING, audio_on=True) is False
    # audio ВЫКЛ — слуховой партии нет, слот занимает штатно
    assert gates.counts_toward_wip("review", CONTENT, AUDIO_PENDING, audio_on=False) is True
    # фраза с теми же modes слух не «ждёт» — слот занимает
    assert gates.counts_toward_wip("review", PHRASE, AUDIO_PENDING, audio_on=True) is True


# ── exam_gate_open: пороги пачки и тормоз ────────────────────────────────────
@pytest.mark.parametrize("pack_n,had_cert,throttled,expected", [
    (gates.PACK_FIRST - 1, False, False, False),
    (gates.PACK_FIRST, False, False, True),       # первый порог
    (gates.PACK_FIRST, True, False, False),       # после сертификации порог выше
    (gates.PACK - 1, True, False, False),
    (gates.PACK, True, False, True),
    (0, False, True, True),                       # тормоз аудита закрывает сам по себе
])
def test_exam_gate_thresholds(pack_n, had_cert, throttled, expected):
    assert gates.exam_gate_open(pack_n, had_cert=had_cert, throttled=throttled) is expected


# ── new_words_blocked / empty_session_reason ─────────────────────────────────
def test_new_words_blocked():
    assert gates.new_words_blocked(gates.WIP_LIMIT, False) is True
    assert gates.new_words_blocked(0, True) is True
    assert gates.new_words_blocked(gates.WIP_LIMIT - 1, False) is False


def test_empty_reason_priority():
    """Слух → экзамен → перебор → всё повторено; порядок фиксирован."""
    assert gates.empty_session_reason(has_audio_pending=True, gate_open=True,
                                      in_work=99) == "listen_pending"
    assert gates.empty_session_reason(has_audio_pending=False, gate_open=True,
                                      in_work=99) == "exam_pending"
    assert gates.empty_session_reason(has_audio_pending=False, gate_open=False,
                                      in_work=gates.WIP_LIMIT) == "wip_full"
    assert gates.empty_session_reason(has_audio_pending=False, gate_open=False,
                                      in_work=0) == "all_done"
