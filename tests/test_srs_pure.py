"""ЭТАП 2: чистое SRS-ядро (srs/) — матрица видов рампы, контракты шагов, паритет с фасадом.

Ядро не знает про БД: всё тестируется значениями. Особые ветки (критика плана):
grandfathering служебных СИММЕТРИЧЕН в обе стороны рубильника cloze (старая контентная/choice
рампа = выучено при cloze ВКЛ; пройденная cloze-рампа = выучено при cloze ВЫКЛ) — фикстуры явные."""
import pytest

from srs import cells, status, steps
from srs.cells import CONTENT, FUNC_CLOZE, FUNC_CHOICE, PHRASE


# ── ramp_kind: полная матрица ────────────────────────────────────────────────
@pytest.mark.parametrize("phrase,func,cloze,expected", [
    (True, False, True, PHRASE),
    (True, True, True, PHRASE),        # фраза побеждает всё
    (False, True, True, FUNC_CLOZE),
    (False, True, False, FUNC_CHOICE),
    (False, False, True, CONTENT),
    (False, False, False, CONTENT),
])
def test_ramp_kind_matrix(phrase, func, cloze, expected):
    assert cells.ramp_kind(phrase_playable=phrase, function_word=func,
                           cloze_enabled=cloze) == expected


def test_cells_are_immutable_tuples():
    """Раньше «вид рампы» держался на identity списков — теперь наборы иммутабельны,
    и рефактор cells_of не может молча сломать классификацию."""
    for kind in (CONTENT, FUNC_CLOZE, FUNC_CHOICE, PHRASE):
        assert isinstance(cells.cells_of(kind), tuple)


# ── grandfathering: однонаправленный ─────────────────────────────────────────
OLD_CONTENT_DONE = {c: "1" for c in cells.REQUIRED_CELLS}
CLOZE_DONE = {c: "1" for c in cells.FUNC_CELLS}


def test_grandfather_func_word_mastered_by_old_content_ramp():
    """Служебное, выученное СТАРОЙ контентной рампой, остаётся выученным при cloze ВКЛ."""
    assert status.is_mastered(FUNC_CLOZE, OLD_CONTENT_DONE) is True


def test_reverse_grandfather_cloze_to_choice():
    """Обратный флип рубильника (слово сдало cloze, затем cloze ВЫКЛючили → рампа «только выбор»):
    пройденная cloze-рампа ГРАНДФАЗИТ FUNC_CHOICE — слово остаётся выученным, откат рубильника его
    не «разучивает» (симметрично forward-грандфазингу; cloze тяжелее выбора). Частично сданная
    cloze-рампа — ещё НЕ выучено."""
    assert status.is_mastered(FUNC_CHOICE, CLOZE_DONE) is True
    assert status.is_mastered(FUNC_CHOICE, {c: "1" for c in cells.FUNC_CELLS_CHOICE}) is True
    assert status.is_mastered(FUNC_CHOICE, {cells.FUNC_CELLS[0]: "1"}) is False


def test_grandfather_func_word_mastered_by_choice_ramp():
    """При ВКЛючении cloze служебное, доученное упрощённой рампой «только выбор» (FUNC_CHOICE,
    cloze был выключен), остаётся выученным — включение банка не воскрешает доученные служебные
    (иначе у активного юзера всплыла бы куча «готовых» слов). choice-клетки ⊂ REQUIRED_CELLS."""
    choice_done = {c: "1" for c in cells.FUNC_CELLS_CHOICE}
    assert status.is_mastered(FUNC_CLOZE, choice_done) is True
    # частично сданная choice-рампа (только одна клетка) — ещё НЕ выучено
    one_cell = {cells.FUNC_CELLS_CHOICE[0]: "1"}
    assert status.is_mastered(FUNC_CLOZE, one_cell) is False


# ── audio-pending и контракт review_step (инциденты #3/#4) ───────────────────
AUDIO_PENDING = {c: "1" for c in cells.REQUIRED_CELLS if c != cells.AUDIO_CELL}


def test_audio_pending_only_for_content():
    assert status.is_audio_pending(CONTENT, AUDIO_PENDING) is True
    assert status.is_audio_pending(CONTENT, OLD_CONTENT_DONE) is False
    for kind in (FUNC_CLOZE, FUNC_CHOICE, PHRASE):
        assert status.is_audio_pending(kind, AUDIO_PENDING) is False


def test_review_step_refuses_audio_pending_word():
    """КОНТРАКТ: слуховое слово не выдаётся на текстовый повтор — guard внутри ядра,
    любой будущий фолбэк-пул физически не может его утащить (инцидент #4)."""
    assert steps.review_step(CONTENT, AUDIO_PENDING, audio_on=True) is None
    # audio ВЫКЛ — слуховой партии нет, повтор штатный
    assert steps.review_step(CONTENT, AUDIO_PENDING, audio_on=False) == \
        ("input_int2no", "input", "int2no")
    # выученное целиком — повтор штатный и при audio ВКЛ
    assert steps.review_step(CONTENT, OLD_CONTENT_DONE, audio_on=True) == \
        ("input_int2no", "input", "int2no")


def test_next_step_defers_audio_cell():
    """audio ВКЛ: текстовые ступени идут вперёд, осталась только аудио-клетка → None
    («ждёт слух»); audio ВЫКЛ: аудио-клетка идёт текстом по порядку."""
    m = {"choice_int2no": "1"}
    assert steps.next_step(CONTENT, m, attempts=2, audio_on=True) == \
        ("build_int2no", "build", "int2no")
    assert steps.next_step(CONTENT, m, attempts=2, audio_on=False) == \
        ("choice_no2int", "choice", "no2int")
    assert steps.next_step(CONTENT, AUDIO_PENDING, attempts=5, audio_on=True) is None
    assert steps.next_step(CONTENT, {}, attempts=0, audio_on=True) == ("card", "study", None)


# ── паритет фасада db.learning со старым поведением ─────────────────────────
async def test_facade_parity(fresh_db):
    """Фасадные функции db.learning дают те же ответы, что и до выноса ядра."""
    from db import learning as L
    row = {"norwegian": "hus", "data": '{"part_of_speech": "noun"}',
           "correct": 2, "incorrect": 0, "strength": 50, "reps": 2,
           "known": None, "archived": None}
    assert L.required_cells(row) == cells.REQUIRED_CELLS
    assert L._is_mastered(row, OLD_CONTENT_DONE) is True
    assert L.status_of(row, OLD_CONTENT_DONE) == "mastered"
    assert L._next_step(row, {"choice_int2no": "1"}, audio_on=False) == \
        ("choice_no2int", "choice", "no2int")
    assert L._review_step(row, AUDIO_PENDING, audio_on=True) is None
    assert L._is_audio_pending(row, AUDIO_PENDING, {"part_of_speech": "noun"}) is True
