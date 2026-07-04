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


def empty_session_reason(*, has_audio_pending: bool, gate_open: bool, in_work: int,
                         wip_limit: int = WIP_LIMIT) -> str:
    """Честная причина пустой сессии для фронта (порядок приоритета фиксирован):
    слуховая партия ждёт → экзамен пачки → перебор в работе → всё повторено."""
    if has_audio_pending:
        return "listen_pending"
    if gate_open:
        return "exam_pending"
    if in_work >= wip_limit:
        return "wip_full"
    return "all_done"
