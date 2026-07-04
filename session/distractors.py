"""Дистракторы choice-элементов как чистый патч (Этап 7).

Раньше _attach_choice_options (db/learning.py) сам ходил в embcache и БД и
мутировал элементы по фильтру «не grammar». Теперь: оркестратор приносит
соседей (neighbors) и их слова (words) данными → options_patch чисто считает
{pool_id: {options, distractors}} → apply_patch применяет ТОЛЬКО контентным
choice-элементам. Морфологические варианты грамм-тира (own_options=True)
перезаписать невозможно by construction.
"""


def _has_own_options(e):
    # Переходная мягкость (критика Этапа 7): у элементов, построенных до
    # make_element (нет ключа own_options), решает старый тег grammar —
    # «тихий фолбэк», а не KeyError на легаси-элементе.
    return bool(e.get("own_options", e.get("grammar", False)))


def choice_targets(session):
    """Элементы, которым нужны ВНЕШНИЕ варианты: mode=choice без своих options."""
    return [e for e in session if e.get("mode") == "choice" and not _has_own_options(e)]


def _answer_of(d, no, direction, lang):
    # до ДВУХ вариантов ответа: (основной, второй|None)
    if direction == "int2no":
        return (no, None)
    tr = ((d or {}).get("translate", {}) or {}).get(lang) or []
    return (tr[0] if tr else None, tr[1] if len(tr) > 1 else None)


def options_patch(items, *, neighbors, words, lang, n=3):
    """{pool_id: {"options": [{w,alt}], "distractors": [w]}}. Вход НЕ мутируется.

    neighbors — {pool_id: [id соседей по убыванию близости]} (embcache),
    words — {id: {"norwegian", "data"}} слова-кандидаты батчем."""
    patch = {}
    for e in items:
        direction = e.get("direction") or "int2no"
        no = e.get("no") or ""
        tr_all = e.get("translate", {}) or {}
        # СМЫСЛ цели = все её переводы на язык юзера. Дистрактор-СИНОНИМ (его переводы
        # пересекаются со смыслом цели) исключаем — иначе вариант оказался бы тоже верным
        # и вопрос нечестным (avstand=[расстояние,дистанция] vs distanse=[дистанция,…]).
        target_mean = {x.strip().lower() for x in (tr_all.get(lang) or []) if x}
        # плюс не повторяем сами допустимые ответы (все переводы / норв. формы цели)
        own = ({(no or "").strip().lower()} | {x.strip().lower() for x in (tr_all.get("no") or []) if x}) \
            if direction == "int2no" else set(target_mean)
        own.discard("")
        nb_ids = neighbors.get(e["pool_id"]) or []   # уже по убыванию близости
        ordered = [words[i] for i in nb_ids if i in words]
        out, seen = [], set(own)
        for c in ordered:
            cd = c.get("data") or {}
            cmean = {x.strip().lower() for x in ((cd.get("translate", {}) or {}).get(lang) or []) if x}
            if target_mean and (cmean & target_mean):   # синоним по смыслу — не годится
                continue
            a, alt = _answer_of(cd, c.get("norwegian"), direction, lang)
            la = (a or "").strip().lower()
            if a and la not in seen:
                out.append({"w": a, "alt": alt})
                seen.add(la)
            if len(out) >= n:
                break
        patch[e["pool_id"]] = {"options": out, "distractors": [o["w"] for o in out]}
    return patch


def apply_patch(session, patch):
    """Применить патч к сессии: только контентные choice-элементы; свои options
    (грамм-тир) не перезаписываются никогда."""
    for e in session:
        p = patch.get(e.get("pool_id"))
        if p is not None and e.get("mode") == "choice" and not _has_own_options(e):
            e.update(p)
