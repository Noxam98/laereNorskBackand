"""Статусы слова: выучено / ждёт слух / рабочий статус. Чистые функции от (kind, modes, …)."""
from .cells import (CONTENT, FUNC_CLOZE, FUNC_CHOICE, PHRASE, AUDIO_CELL, REQUIRED_CELLS,
                    FUNC_CELLS, FUNC_CELLS_CHOICE, PHRASE_CELLS_LEGACY, cells_of)


def is_mastered(kind: str, modes: dict | None) -> bool:
    """Выучено: все клетки рампы == '1'. Grandfathering СЛУЖЕБНЫХ симметричен в ОБЕ стороны
    переключения рубильника CLOZE_ENABLED — слово, доведённое ЛЮБОЙ прежней служебной рампой,
    остаётся выученным при смене режима (не «разучивается» и не «воскресает пачкой»):
      • cloze ВКЛючили (kind FUNC_CLOZE): засчитываем старую контентную/choice-рампу
        (choice-клетки ⊂ REQUIRED_CELLS, один чек покрывает обе); включение банка не воскрешает
        доученные служебные;
      • cloze ВЫКЛючили (kind FUNC_CHOICE, kill-switch): засчитываем пройденную cloze-рампу
        (FUNC_CELLS) — cloze тяжелее выбора, сдавший cloze тем более знает слово; иначе откат
        рубильника «разучивал» бы cloze-выученные (хранимый mastered=1 при этом и так остаётся).

    Grandfathering ФРАЗ (добавили ступень build_int2no между «порядком слов» и вводом):
    прошедший СТАРУЮ рампу целиком (PHRASE_CELLS_LEGACY, финал — свободный ввод по памяти)
    остаётся выученным. Сборка из букв легче ввода, поэтому новая ступень не должна воскрешать
    уже сданные фразы пачкой."""
    m = modes or {}
    if all(m.get(c, "") == "1" for c in cells_of(kind)):
        return True
    if kind == FUNC_CLOZE and all(m.get(c, "") == "1" for c in FUNC_CELLS_CHOICE):
        return True
    if kind == FUNC_CHOICE and all(m.get(c, "") == "1" for c in FUNC_CELLS):
        return True
    if kind == PHRASE and all(m.get(c, "") == "1" for c in PHRASE_CELLS_LEGACY):
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
