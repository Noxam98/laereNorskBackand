"""Морфология прилагательных (bokmål): согласование (нейтрум/мн.), степени, детектор, дистракторы."""
from ._common import VOWELS, _norm, _syllables, _degeminate_tail, _plausible


# Супплетив/умлаут степеней — НЕ выводим правилом, всегда нерегулярны.
# {лемма: (comparative, superlative)}
ADJ_SUPPLETIVE = {
    "god": ("bedre", "best"),
    "vond": ("verre", "verst"),
    "ond": ("verre", "verst"),
    "gammel": ("eldre", "eldst"),
    "stor": ("større", "størst"),
    "ung": ("yngre", "yngst"),
    "lang": ("lengre", "lengst"),
    "tung": ("tyngre", "tyngst"),
    "liten": ("mindre", "minst"),
    "mange": ("flere", "flest"),
    "mye": ("mer", "mest"),
    "få": ("færre", "færrest"),
    "bra": ("bedre", "best"),
}

# Прилагательные с НЕРЕГУЛЯРНЫМИ ПОЗИТИВ-формами (нейтрум/мн.ч. не выводятся
# правилом): liten (lite/små), annen (annet/andre), egen (eget/egne).
# Их позитив-предикторы возвращают None.
ADJ_IRREGULAR_POSITIVE = {"liten", "annen", "egen"}

# Полностью особые/супплетивные (для общего флага is_irregular).
ADJ_EXPLICIT_IRREGULAR = (
    {"liten", "annen", "egen", "bra"} | set(ADJ_SUPPLETIVE)
)

# Несклоняемые цвета/заимствования: НЕ +t, НЕ +e.
ADJ_INDECLINABLE = {
    "rosa", "lilla", "beige", "oransje", "turkis", "stille", "gratis",
    "felles", "stakkars", "diverse", "ekstra", "bra",
}

# -sk прилагательные БЕЗ нейтрального -t, хотя односложные (национальности/языки):
# norsk, svensk, dansk... + они же часто несравнимы морфологически.
ADJ_SK_NO_T = {
    "norsk", "svensk", "dansk", "tysk", "fransk", "engelsk", "spansk",
    "russisk", "finsk", "polsk", "gresk", "barnslig",  # barnslig — на -ig, дубль ок
}

# Синкопирующие основы -el/-en/-er (plural/comp/sup с выпадением -e-, дегеминацией).
# Нейтрум — БЕЗ синкопы (gammel→gammelt, vakker→vakkert).
ADJ_SYNCOPE = {
    "gammel", "vakker", "sulten", "åpen", "enkel", "voksen", "sikker",
    "egen", "moden", "edel", "nø9", "doven", "gylden", "ærlig9",
}
ADJ_SYNCOPE.discard("nø9")
ADJ_SYNCOPE.discard("ærlig9")


def regular_neuter(adj):
    """Нейтрум (intetkjønn). None если не уверены."""
    w = _norm(adj)
    if not w:
        return None
    if w in ADJ_INDECLINABLE or w in ADJ_IRREGULAR_POSITIVE:
        return None

    # уже на -e: несклоняемо в нейтруме (øde, ekte, bedre)
    if w.endswith("e"):
        return w
    # -ig: нет +t (viktig→viktig)
    if w.endswith("ig"):
        return w
    # -sk: национальности/языки и многосложные (-isk: praktisk) БЕЗ +t;
    #      односложные «качественные» -sk берут +t (frisk→friskt, fersk→ferskt).
    # NB: -isk как СУФФИКС только в многосложных, иначе frisk попадёт ложно.
    if w.endswith("sk"):
        if w in ADJ_SK_NO_T or _syllables(w) >= 2:
            return w
        return w + "t"
    # причастные -et: без +t (forelsket)
    if w.endswith("et"):
        return w
    # уже на -t: без +t (svart, kort)
    if w.endswith("t"):
        return w
    # упрощение -nn→-nt (grønn→grønt, tynn→tynt, sann→sant) — почти 100% регулярно
    if w.endswith("nn"):
        return w[:-1] + "t"
    # удвоение -tt у односложных на ОДИНОЧНЫЙ ударный долгий гласный
    # (ny→nytt, blå→blått, fri→fritt). НЕ срабатывает на дифтонге (grei→greit):
    # нужен ровно один финальный гласный, перед ним согласная (или начало слова).
    if (
        _syllables(w) == 1
        and w[-1] in VOWELS
        and (len(w) == 1 or w[-2] not in VOWELS)
    ):
        return w + "tt"
    # общий случай
    return w + "t"


def _adj_syncope_stem(w):
    """Основа синкопирующего прил.: gammel→gaml, vakker→vakr, enkel→enkl."""
    base = _degeminate_tail(w[:-2])  # 'gamm'→'gam'
    return base + w[-1]              # + 'l' → 'gaml'


def regular_plural(adj):
    """Мн.ч./опр. (слабое склонение) = +e. None если не уверены."""
    w = _norm(adj)
    if not w:
        return None
    if w in ADJ_INDECLINABLE or w in ADJ_IRREGULAR_POSITIVE:
        return None
    if w.endswith("e"):
        return w                    # øde→øde
    # причастные -et: мн.ч. = лемме (forelsket) или книжн. -ede; каноник = лемма.
    if w.endswith("et") and _syllables(w) >= 2:
        return w
    if w in ADJ_SYNCOPE and w.endswith(("el", "en", "er")):
        return _adj_syncope_stem(w) + "e"   # gammel→gamle, vakker→vakre
    return w + "e"                  # fin→fine, stygg→stygge


def regular_comparative(adj):
    """Сравнительная = +ere. None если несравнимо/супплетив."""
    w = _norm(adj)
    if not w:
        return None
    if w in ADJ_SUPPLETIVE or w in ADJ_EXPLICIT_IRREGULAR or w in ADJ_INDECLINABLE:
        return None
    if w.endswith("e"):
        return None                 # на -e обычно несравнимо/описательно
    if w in ADJ_SYNCOPE and w.endswith(("el", "en", "er")):
        return _adj_syncope_stem(w) + "ere"   # enkel→enklere, vakker→vakrere
    return w + "ere"                # fin→finere, billig→billigere


def regular_superlative(adj):
    """Превосходная = +est (-ig → +st). None если несравнимо/супплетив."""
    w = _norm(adj)
    if not w:
        return None
    if w in ADJ_SUPPLETIVE or w in ADJ_EXPLICIT_IRREGULAR or w in ADJ_INDECLINABLE:
        return None
    if w.endswith("e"):
        return None
    if w.endswith("ig"):
        return w + "st"             # billig→billigst, viktig→viktigst
    if w in ADJ_SYNCOPE and w.endswith(("el", "en", "er")):
        return _adj_syncope_stem(w) + "est"   # enkel→enklest
    return w + "est"                # fin→finest


def is_irregular_adj(adj, forms):
    """Детектор нерегулярности прил. forms: {neuter,plural,comparative,superlative}.

    Возвращает (bool, reason). is_irregular = OR по клеткам ИЛИ явный список.
    """
    w = _norm(adj)

    if w in ADJ_EXPLICIT_IRREGULAR or w in ADJ_SUPPLETIVE:
        return True, "explicit_list"

    # Несклоняемое: позитив-формы = лемме → НЕ нерегулярно (нечего дрилить),
    # но и регуляр не применяем.
    if w in ADJ_INDECLINABLE:
        return False, "indeclinable"

    n = _norm(forms.get("neuter"))
    pl = _norm(forms.get("plural"))
    comp = _norm(forms.get("comparative"))
    sup = _norm(forms.get("superlative"))

    # neuter
    if n:
        pred = regular_neuter(adj)
        if pred is not None and n != pred:
            return True, "neuter_mismatch"

    # plural
    if pl:
        pred = regular_plural(adj)
        if pred is not None and pl != pred:
            return True, "plural_mismatch"

    # comparative: учесть -st/-est дублет и описательное 'mer ...'
    if comp:
        if " " in comp:             # 'mer norsk' — описательное, несравнимо
            pass                    # не нерегуляр по морфологии
        else:
            pred = regular_comparative(adj)
            if pred is not None and comp != pred:
                return True, "comparative_mismatch"

    # superlative: принять и -st, и -est (терпимость к -ig дублету)
    if sup and " " not in sup:
        pred = regular_superlative(adj)
        if pred is not None and sup != pred:
            # альтернативный суффикс не считаем нерегулярным
            alt = None
            base = regular_plural(adj)  # как proxy основы
            if pred.endswith("est") and sup == pred[:-3] + "st":
                alt = sup
            elif pred.endswith("st") and sup == pred[:-2] + "est":
                alt = sup
            if alt is None:
                return True, "superlative_mismatch"

    return False, "regular"


# ── Опции для упражнения-РАЗЛИЧЕНИЯ формы прилагательного ─────────────────────
# Клетки форм прил. в треке форм. Компаратив/суперлатив могут отсутствовать
# (несравнимые/описательные mer/mest) — тогда correct пуст и клетку не дрилим.
ADJ_FORM_CELLS = ("neuter", "plural", "comparative", "superlative")


def _adj_regular(adj, cell):
    return {"neuter": regular_neuter, "plural": regular_plural,
            "comparative": regular_comparative, "superlative": regular_superlative}[cell](adj)


def adj_form_options(adj, forms, cell, n=3):
    """(correct, [distractors]) для клетки-различения формы прилагательного.
    Дистракторы: наивная регулярная форма (для нерегулярных — «ожидаемая, но неверная»:
    god→*godere, stor→*storere, grønn→*grønnt) + РЕАЛЬНЫЕ соседние формы (позитив/нейтрум/мн./степени)
    + лемма (позитив). Валидная регулярная форма и малформы отфильтрованы.

    Для нерегулярных regular_* → None, значит наивная форма НЕ попадает в «допустимые» и работает
    дистрактором. Для регулярных наивная = correct и отсеивается, дистракторы дают соседние формы."""
    w = _norm(adj)
    forms = forms or {}
    correct = _norm(forms.get(cell))
    if not correct:
        return None, []

    reg = _adj_regular(adj, cell)
    allowed = {correct}
    if reg:
        allowed.add(reg)

    neu = _norm(forms.get("neuter"))
    pl = _norm(forms.get("plural"))
    comp = _norm(forms.get("comparative"))
    sup = _norm(forms.get("superlative"))

    # наивная регулярная форма для этой клетки (слепое правило без синкопы/умлаута/дифтонга)
    naive = {"neuter": w + "t", "plural": w + "e",
             "comparative": w + "ere", "superlative": w + "est"}[cell]

    # порядок: наивная ошибка + «ближайшая» соседняя форма первыми (максимум педагогики)
    order = {
        "neuter":      [naive, w, pl, comp, sup],       # позитив↔нейтрум — главная путаница
        "plural":      [naive, neu, w, comp, sup],
        "comparative": [naive, sup, neu, pl, w],        # компаратив↔суперлатив
        "superlative": [naive, comp, neu, pl, w],
    }[cell]

    if cell == "neuter":
        fallback = [w + "t", w + "tt", w + "e"]
    elif cell == "plural":
        fallback = [w + "e", w + "ere"]
    elif cell == "comparative":
        fallback = [w + "ere", w + "est"]
    else:  # superlative
        fallback = [w + "est", w + "st", w + "ere"]

    seen, out = set(), []
    for f in order + fallback:
        if not f or f in allowed or f in seen or not _plausible(f):
            continue
        seen.add(f)
        out.append(f)
        if len(out) >= n:
            break
    return correct, out


# ── Дистракторы для прилагательного ──────────────────────────────────────────

def distractors_adj(adj, forms):
    """Правдоподобные неверные формы прил. для клеток выбора."""
    w = _norm(adj)
    res = {"neuter": [], "plural": [], "comparative": [], "superlative": []}

    c_n = _norm(forms.get("neuter"))
    c_pl = _norm(forms.get("plural"))
    c_comp = _norm(forms.get("comparative"))
    c_sup = _norm(forms.get("superlative"))

    # neuter: слепое +t (или сохранить/убрать удвоение).
    naive_n = set()
    naive_n.add(w + "t")                    # viktig→*viktigt, norsk→*norskt
    if _syllables(w) == 1 and w[-1] in VOWELS:
        naive_n.add(w + "t")                # ny→*nyt (одно t)
    if w.endswith("nn"):
        naive_n.add(w + "t")                # grønn→*grønnt (сохранить nn)
    res["neuter"] = [f for f in naive_n if f and f != c_n]

    # plural: без синкопы / ложное +e.
    naive_pl = set()
    naive_pl.add(w + "e")                   # gammel→*gammele, rosa→*rosae
    if w in ADJ_SYNCOPE:
        naive_pl.add(w + "e")
    res["plural"] = [f for f in naive_pl if f and f != c_pl]

    # comparative: наивное +ere без умлаута/синкопы.
    naive_comp = {w + "ere"}                # stor→*storere, god→*godere
    if c_sup:                               # чужой суффикс из суперлатива
        pass
    res["comparative"] = [f for f in naive_comp if f and f != c_comp]

    # superlative: наивное +est / чужой -st↔-est.
    naive_sup = {w + "est", w + "st"}       # billig→*billigest, fin→*finst
    res["superlative"] = [f for f in naive_sup if f and f != c_sup]

    return res
