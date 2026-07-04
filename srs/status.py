"""Статусы слова: выучено / ждёт слух / рабочий статус. Чистые функции от (kind, modes, …)."""
from .cells import CONTENT, FUNC_CLOZE, AUDIO_CELL, REQUIRED_CELLS, FUNC_CELLS_CHOICE, cells_of


def is_mastered(kind: str, modes: dict | None) -> bool:
    """Выучено: все клетки рампы == '1'. Grandfathering служебных ПРИ ВКЛЮЧЕНИИ cloze:
    слово, уже доведённое ЛЮБОЙ прежней рампой — старой контентной (choice/build/input) ИЛИ
    упрощённой «только выбор» (FUNC_CHOICE при выключенном cloze) — остаётся выученным, включение
    cloze не воскрешает «доученные» служебные (иначе у активного юзера всплыла бы куча слов).
    choice-клетки ⊂ REQUIRED_CELLS, поэтому один их чек покрывает и старую контентную рампу.
    Обратно НЕ грандфазируем: cloze-прогресс не засчитывается за choice-клетки (FUNC_CHOICE
    считается только по своим клеткам — см. тест no_reverse_grandfather)."""
    m = modes or {}
    if all(m.get(c, "") == "1" for c in cells_of(kind)):
        return True
    if kind == FUNC_CLOZE and all(m.get(c, "") == "1" for c in FUNC_CELLS_CHOICE):
        return True
    return False


def is_audio_pending(kind: str, modes: dict | None) -> bool:
    """Текстовая рампа сдана, аудио-клетка нет → слово «ждёт слух» (живёт в слуховой
    партии, НЕ занимает слоты дневной сессии — инцидент #3 3.07). Только контентные:
    у фраз/служебных choice_no2int — текстовый выбор, слух они не «ждут»."""
    if kind != CONTENT:
        return False
    m = modes or {}
    return m.get(AUDIO_CELL, "") != "1" and all(
        m.get(c, "") == "1" for c in REQUIRED_CELLS if c != AUDIO_CELL)


def word_status(kind: str, modes: dict | None, *, attempts: int, strength: float,
                incorrect: int, reps: int, known: bool, archived: bool) -> str:
    """Рабочий статус слова для логики сессии (не для отображения)."""
    if known:
        return "known"
    if archived:
        return "archived"
    if attempts == 0:
        return "new"
    if is_mastered(kind, modes):
        return "mastered"
    if strength < 40 and incorrect >= 2:
        return "weak"
    if reps >= 1:
        return "review"
    return "learning"
