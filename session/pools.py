"""Источники кандидатов дневной сессии: приоритетные пулы, отбор со степами,
кулдаун «умной очереди», досрочные повторы. Чистые функции (Этап 4).

Вход — enriched-элементы (dict: row/modes/status/due/data), предикаты и степ-функции
приходят параметрами: модуль не знает ни про БД, ни про настройки юзера.
"""


def attempts_of(e) -> int:
    rr = e["row"]
    return (rr.get("correct") or 0) + (rr.get("incorrect") or 0)


def build_pools(enriched, *, is_certified, is_function_word):
    """Приоритетные пулы кандидатов. Порядок использования фиксирован:
    returned → overdue → due_mastered(повтор) → weak → maturing."""
    returned = sorted(
        [e for e in enriched if e["status"] in ("weak", "learning", "review")
         and (e["row"].get("lapses") or 0) > 0],
        key=lambda e: (e["row"].get("strength") or 0, e["row"].get("due_at") or ""))
    overdue = sorted([e for e in enriched if e["due"]],
                     key=lambda e: e["row"].get("due_at") or "")
    weak = sorted([e for e in enriched if e["status"] == "weak"],
                  key=lambda e: e["row"].get("strength") or 0)
    # дозревающие: не выучено и не в архиве; тронутые вперёд, совсем новые в хвост
    maturing = sorted(
        [e for e in enriched if e["status"] in ("new", "learning", "review")],
        key=lambda e: (attempts_of(e) == 0, e["row"].get("strength") or 0))
    # выученные не-сертифицированные контентные с наступившим due → текстовый повтор
    # (сертифицированные — свой аудит; служебные — cloze-рампа, повтора так нет)
    due_mastered = sorted(
        [e for e in enriched if e["status"] == "mastered" and e["due"]
         and not is_certified(e["row"])
         and not is_function_word(e["row"]["norwegian"], e["data"])],
        key=lambda e: e["row"].get("due_at") or "")
    return [(returned, False), (overdue, False), (due_mastered, True),
            (weak, False), (maturing, False)]


def select_candidates(pools, *, next_step, review_step, func_locked):
    """Отбор кандидатов из пулов по приоритету: дедуп по pool_id и по омографу
    (norwegian, pos); mastered — только в review-пул; новое служебное придерживается
    пословным порогом; слову назначается ступень (нет ступени — слово не идёт)."""
    cand, seen, seen_wp = [], set(), set()
    for pool, review in pools:
        for e in pool:
            pid = e["row"]["pool_id"]
            wp = (e["row"]["norwegian"], (e["data"] or {}).get("part_of_speech", "") or "")
            if pid in seen or wp in seen_wp or e["status"] in ("archived", "known"):
                continue
            if not review and e["status"] == "mastered":
                continue
            if attempts_of(e) == 0 and func_locked(e):
                continue
            step = review_step(e) if review else next_step(e)
            if not step:
                continue
            seen.add(pid)
            seen_wp.add(wp)
            cand.append((e, step))
    return cand


def apply_cooldown(cand, *, cooldown_cut: str):
    """«Умная очередь»: только что показанные (last_seen ≥ cooldown_cut) — в хвост,
    НЕ выбрасываются; относительный порядок внутри групп сохранён."""
    def fresh(c):
        ls = c[0]["row"].get("last_seen")
        return bool(ls) and ls >= cooldown_cut
    return [c for c in cand if not fresh(c)] + [c for c in cand if fresh(c)]


def early_review_pool(enriched, *, review_step, size: int):
    """Досрочные повторы (анти-тупик ветерана): начатые слова с ближайшим due.
    Слуховые слова не утянет — review_step ядра сам возвращает для них None."""
    early = sorted(
        [e for e in enriched if e["status"] in ("learning", "review", "weak")
         and attempts_of(e) > 0 and not e["row"].get("archived") and not e["row"].get("known")],
        key=lambda e: e["row"].get("due_at") or "~")
    out = []
    for e in early:                      # ФИЛЬТРУЕМ (review_step) → ПОТОМ режем: у audio-юзера
        step = review_step(e)            # слуховые слова дают None, а срез early[:size] ДО фильтра
        if step:                         # оставлял бы пул пустым (анти-тупик не срабатывал).
            out.append((e, step))
            if len(out) >= size:
                break
    return out
