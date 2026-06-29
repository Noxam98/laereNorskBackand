"""cloze можно временно выключить (CLOZE_ENABLED, дефолт ВЫКЛ): служебные слова идут «только выбор»,
без задания «вставь пропущенное». Вернуть — Fly secret CLOZE_ENABLED=1."""
import json
from db.learning import required_cells, CLOZE_ENABLED, REQUIRED_CELLS


def test_cloze_disabled_by_default():
    assert CLOZE_ENABLED is False


def test_function_word_choice_only_when_cloze_off():
    """Служебное слово («å») при выключенном cloze → рампа «только выбор», без cloze-ступеней."""
    cells = required_cells({"norwegian": "å", "data": "{}"})
    assert "cloze_1" not in cells
    assert cells == ["choice_int2no", "choice_no2int"]


def test_conjunction_choice_only_when_cloze_off():
    cells = required_cells({"norwegian": "og", "data": json.dumps({"part_of_speech": "conjunction"})})
    assert all(not c.startswith("cloze") for c in cells)


def test_content_word_ramp_unchanged():
    """Контентное слово — полная рампа choice/build/input, как и было."""
    cells = required_cells({"norwegian": "hus", "data": json.dumps({"part_of_speech": "noun"})})
    assert cells == REQUIRED_CELLS
