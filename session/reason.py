"""Диагностика composition.reason — «почему сессия такая» (Этап 8). Чистая.

Приоритет — ДАННЫМИ: упорядоченный список (код, предикат), а не хрупкий if/elif,
размазанный по build_session. Порядок и состав — контракт фронта (кнопка старта):
    early_review   — сессия ЕСТЬ, но собрана из досрочных повторов (анти-тупик)
    listen_pending — пусто: слова ждут слуховой сдачи (инцидент #4)
    exam_pending   — пусто: ворота закрыты на экзамен пачки
    wip_full       — пусто: лимит «в работе» перебран
    all_done       — пусто: всё повторено, due нет
Единственный дом лестницы: бывший srs.gates.empty_session_reason перенесён сюда
(два дома = дрейф критериев, ровно тот класс багов, который выпиливаем).
"""

RULES = (
    ("early_review",   lambda c: c["session_len"] > 0 and c["early_review"]),
    ("listen_pending", lambda c: c["session_len"] == 0 and c["has_audio_pending"]),
    ("exam_pending",   lambda c: c["session_len"] == 0 and c["gate_open"]),
    ("wip_full",       lambda c: c["session_len"] == 0 and c["in_work"] >= c["wip_limit"]),
    ("all_done",       lambda c: c["session_len"] == 0),
)


def session_reason(*, session_len, scoped, early_review, has_audio_pending,
                   gate_open, in_work, wip_limit):
    """Код причины для composition.reason или None (обычная непустая сессия).
    Дрилл по набору (scoped) — вне диагностики дневного конвейера: всегда None."""
    if scoped:
        return None
    ctx = {"session_len": session_len, "early_review": early_review,
           "has_audio_pending": has_audio_pending, "gate_open": gate_open,
           "in_work": in_work, "wip_limit": wip_limit}
    for code, pred in RULES:
        if pred(ctx):
            return code
    return None
