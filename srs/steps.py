"""Шаги рампы: следующая невыполненная ступень и шаг повтора. Чистые функции.

КОНТРАКТ (инцидент #4, 3.07): review_step сам возвращает None для слова, ждущего
слуховой сдачи, — защита живёт ВНУТРИ функции, а не в вежливости вызывающего.
Любой будущий фолбэк-пул физически не может утащить слуховое слово в текстовый повтор.
"""
from .cells import CONTENT, AUDIO_CELL, cells_of, ramp_cells
from .status import is_audio_pending, is_mastered


def next_step(kind: str, modes: dict | None, *, attempts: int, audio_on: bool,
              choice_on: bool = True):
    """Следующая ступень рампы → (cell, mode, direction) | None.
    Совсем новое (0 попыток) → пассивная карточка. audio_on=True → у контентных слов
    аудио-клетка пропускается (уходит в слуховую партию); остаётся только она → None
    («ждёт слух», в дневную сессию не идёт). mastered → None.
    choice_on=False → ступень выбора выключена тумблером и в выдаче не участвует; её клетки
    засчитывает сдача более сложной ступени (см. cells.skipped_choice).

    Второй проход по ПОЛНОЙ рампе — анти-тупик: если эффективная рампа исчерпана, а слово не
    mastered (битое состояние вроде build='1' при пустом choice), слово всё равно получает
    свою клетку и доучивается, а не зависает навсегда в «в работе»."""
    if attempts == 0:
        return ("card", "study", None)
    if is_mastered(kind, modes):
        return None
    m = modes or {}
    eff = ramp_cells(kind, choice_on=choice_on, audio_on=audio_on)
    full = cells_of(kind)
    for cells in ((eff,) if eff == full else (eff, full)):
        for cell in cells:
            if audio_on and kind == CONTENT and cell == AUDIO_CELL:
                continue
            if m.get(cell, "") != "1":
                mode, direction = cell.split("_", 1)
                return (cell, mode, direction)
    return None


def review_step(kind: str, modes: dict | None = None, *, audio_on: bool = False):
    """Шаг ПОВТОРА: последняя продуктивная (не аудио) ступень рампы.
    Слово, ждущее слуховой сдачи, на текстовый повтор НЕ выдаётся (None) —
    это контракт функции, а не фильтр вызывающего."""
    if audio_on and is_audio_pending(kind, modes):
        return None
    cells = cells_of(kind)
    if kind == CONTENT:                  # AUDIO_CELL (choice_no2int) — «слух» ТОЛЬКО у контентных;
        cells = tuple(c for c in cells if c != AUDIO_CELL) or cells   # у FUNC_CHOICE/фраз это ТЕКСТ,
    last = cells[-1]                     # его нельзя срезать, иначе повтор скатывается в лёгкий int2no
    mode, direction = last.split("_", 1)
    return (last, mode, direction)
