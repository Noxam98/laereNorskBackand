"""Бюджеты и ворота притока новых слов. Чистые функции — каждое правило имеет ИМЯ.

До Этапа 3 in_work/gate_open считались инлайн внутри build_session._load(), а критерий
пачки дублировал логику exams — инциденты #2/#3 (3.07) выросли ровно из безымянных
инлайн-правил, которые правились не во всех копиях сразу.
"""
from .status import is_audio_pending

WIP_LIMIT = 20    # слов одновременно «в работе» (learning/review/weak); сверх — новые не вводим
PACK_FIRST = 50   # порог первой несданной пачки на экзамен (до первой сертификации)
PACK = 100        # порог последующих пачек


def counts_toward_wip(status: str, kind: str, modes: dict | None, *, audio_on: bool) -> bool:
    """Занимает ли слово слот «в работе». Слово, ждущее ТОЛЬКО слуховой сдачи, слот
    НЕ занимает — оно живёт в параллельной слуховой партии (инцидент #3: 20+ слуховых
    слов душили лимит и дневная сессия ветерана пустела)."""
    if status not in ("learning", "review", "weak"):
        return False
    return not (audio_on and is_audio_pending(kind, modes))


def exam_gate_open(pack_n: int, *, had_cert: bool, throttled: bool) -> bool:
    """«Ворота закрыты на экзамен»: несданная пачка выученных достигла порога
    (первая — PACK_FIRST, дальше PACK) ИЛИ действует тормоз аудита. True = новые
    слова не вводим, пока юзер не сдаст экзамен. throttled уже ВКЛЮЧЁН сюда —
    вызывающие не должны проверять его повторно (инвариант из карты)."""
    return pack_n >= (PACK if had_cert else PACK_FIRST) or throttled


def new_words_blocked(in_work: int, gate_open: bool, *, wip_limit: int = WIP_LIMIT) -> bool:
    """Приток новых слов заблокирован: лимит «в работе» ИЛИ ворота экзамена."""
    return in_work >= wip_limit or gate_open

# Диагностика ПУСТОЙ/досрочной сессии (comp.reason) переехала в session/reason.py (Этап 8):
# там она единой лестницей покрывает и early_review, и empty-ветки — один дом критериев.
