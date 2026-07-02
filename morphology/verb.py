"""Морфология глаголов (bokmål): слабые классы, презенс/претерит/перфект, детектор, дистракторы."""
from ._common import VOWELS, _norm, _syllables, _ends_double_cons, _degeminate_tail, _plausible


# Нерегулярный презенс (НЕ +r).
MODAL_PRESENT = {
    "kunne": "kan", "ville": "vil", "skulle": "skal",
    "måtte": "må", "burde": "bør", "tore": "tør", "turde": "tør",
}
IRREG_PRESENT = {
    "være": "er", "gjøre": "gjør", "vite": "vet",
    "ha": "har", "si": "sier", "spørre": "spør",
}

# Сильные/нерегулярные/супплетивные глаголы — past/perfect правилом НЕ выводим.
STRONG_IRREGULAR = {
    "være", "ha", "gjøre", "vite", "si",
    "gå", "stå", "få", "se", "slå", "le", "be", "dø",
    "drikke", "finne", "synge", "springe", "binde", "vinne", "treffe",
    "ligge", "sitte", "komme", "ta", "gi", "bli",
    "skrive", "drive", "bite", "ride", "gripe", "stige", "skyte",
    "gråte", "falle", "holde", "henge", "hjelpe", "sove",
    "legge", "sette", "selge", "fortelle", "spørre", "telle",
    "dra", "fare", "la", "by", "fryse", "velge", "rekke",
    "forstå", "oppleve", "bestemme",  # приставочные — но bestemme/oppleve слабые;
    # оставлены как осознанно-нерег. только если корень сильный:
}
# Уберём заведомо слабые приставочные, попавшие выше по ошибке композиции:
STRONG_IRREGULAR.discard("oppleve")
STRONG_IRREGULAR.discard("bestemme")

# Депоненты на -s (презенс НЕ +r, формы особые).
S_DEPONENT = {"trives", "synes", "undres", "finnes", "lykkes", "møtes"}

# Class-4 короткие основы на ударный гласный (явный список — эвристика ломается на y/i).
SHORT_VOWEL_STEM = {
    "bo": "bodde", "tro": "trodde", "nå": "nådde", "sy": "sydde",
    "bety": "betydde", "gru": "grudde", "snu": "snudde", "fri": "fridde",
    "g:": "", "ro": "rodde", "spy": "spydde", "gni": "gnidde",
}
SHORT_VOWEL_STEM.pop("g:", None)


def predict_present(inf):
    """Презенс регулярного глагола = инфинитив + 'r'. Спецслучаи — по спискам."""
    inf = _norm(inf)
    if inf in MODAL_PRESENT:
        return MODAL_PRESENT[inf]
    if inf in IRREG_PRESENT:
        return IRREG_PRESENT[inf]
    if inf in S_DEPONENT or inf.endswith("s"):
        return inf                 # trives→trives, synes→synes
    return inf + "r"


regular_present = predict_present  # публичный алиас по ТЗ


def _weak_stem(inf):
    """Основа слабого глагола = инфинитив без конечного -e."""
    return inf[:-1] if inf.endswith("e") else inf


def _is_short_vowel_inf(inf):
    """Class-4: односложная лемма на ударный гласный (bo, tro, nå...)."""
    if inf in SHORT_VOWEL_STEM:
        return True
    return (
        bool(inf)
        and not inf.endswith("e")
        and inf[-1] in VOWELS
        and _syllables(inf) == 1
        and len(inf) <= 4
    )


def predict_weak_class(inf):
    """Регулярный СЛАБЫЙ глагол → (class_id, past, perfect) или (None,None,None).

    4 слабых класса:
      1 (-et): кластер/удвоенный согласный после краткого гласного (kaste→kastet);
      2 (-te): глухой смычный/s или сонорный после долгого (spise→spiste);
      3 (-de): звонкий g/v (leve→levde);
      4 (-dde): односложный на ударный гласный (bo→bodde).
    Перед -te/-de удвоенный согласный УПРОЩАЕТСЯ (bygge→bygde, kalle→kalte).
    """
    inf = _norm(inf)
    if not inf:
        return (None, None, None)

    # CLASS 4 — короткие на гласный.
    if _is_short_vowel_inf(inf):
        if inf in SHORT_VOWEL_STEM:
            past = SHORT_VOWEL_STEM[inf]
            return (4, past, "har " + past[:-1])  # bodde→bodd
        return (4, inf + "dde", "har " + inf + "dd")

    stem = _weak_stem(inf)
    if not stem:
        return (None, None, None)

    last = stem[-1]
    double = _ends_double_cons(stem)
    cluster = len(stem) >= 2 and stem[-2] not in VOWELS and last not in VOWELS

    # CLASS 1 (-et): двойной согласный / кластер. НЕ упрощаем перед -et.
    # НО: bygge→bygde (class3 c упрощением) — эвристика угадает class1,
    # детектор это прощает (сравнение со ВСЕМИ классами). Для ПРЕДСКАЗАНИЯ
    # дефолтим в class1, когда форм нет.
    if double or cluster:
        return (1, stem + "et", "har " + stem + "et")

    # CLASS 3 (-de): звонкий g/v.
    if last in ("g", "v"):
        return (3, stem + "de", "har " + stem + "d")

    # CLASS 2 (-te): глухой/сонорный.
    return (2, stem + "te", "har " + stem + "t")


def regular_past(inf):
    """Каноничный регулярный претерит (или None для сильных/модальных)."""
    inf = _norm(inf)
    if inf in STRONG_IRREGULAR or inf in MODAL_PRESENT or inf in S_DEPONENT:
        return None
    return predict_weak_class(inf)[1]


def regular_perfect(inf):
    """Каноничный регулярный перфект 'har ...' (или None)."""
    inf = _norm(inf)
    if inf in STRONG_IRREGULAR or inf in MODAL_PRESENT or inf in S_DEPONENT:
        return None
    return predict_weak_class(inf)[2]


def all_weak_pasts(inf):
    """Множество ДОПУСТИМЫХ слабых претеритов (любой класс) + дублеты -et/-a.

    Учитывает упрощение удвоенного согласного перед -te/-de:
    kalle→{...,'kalte'}, bygge→{...,'bygde'}.
    """
    inf = _norm(inf)
    out = set()
    if _is_short_vowel_inf(inf):
        if inf in SHORT_VOWEL_STEM:
            out.add(SHORT_VOWEL_STEM[inf])
        else:
            out.add(inf + "dde")
        return out
    stem = _weak_stem(inf)
    if not stem:
        return out
    simp = _degeminate_tail(stem)
    # class1: -et / -a (НЕ упрощаем)
    out.update({stem + "et", stem + "a"})
    # class2/3 c упрощением и без
    out.update({stem + "te", stem + "de", simp + "te", simp + "de"})
    return out


def all_weak_perfects(inf):
    """Множество допустимых слабых причастий (для перфекта, без 'har')."""
    inf = _norm(inf)
    out = set()
    if _is_short_vowel_inf(inf):
        if inf in SHORT_VOWEL_STEM:
            out.add(SHORT_VOWEL_STEM[inf][:-1])  # bodde→bodd
        else:
            out.add(inf + "dd")
        return out
    stem = _weak_stem(inf)
    if not stem:
        return out
    simp = _degeminate_tail(stem)
    out.update({stem + "et", stem + "t", stem + "d", simp + "t", simp + "d"})
    return out


def strip_aux(form):
    """Срезать вспомогательный глагол перфекта: 'har '/'er ' → причастие."""
    f = _norm(form)
    for p in ("har ", "er ", "har", "er"):
        if f.startswith(p):
            rest = f[len(p):].strip()
            if rest:
                return rest
    return f


def is_irregular_verb(inf, forms):
    """Детектор нерегулярности глагола. forms: {'present','past','perfect'}.

    Возвращает (bool, reason). is_irregular = OR трёх проверок.
    """
    inf = _norm(inf)

    # 1) Явный список.
    if inf in STRONG_IRREGULAR or inf in MODAL_PRESENT or inf in S_DEPONENT:
        return True, "explicit_list"

    # 2) Презенс.
    present = _norm(forms.get("present"))
    if present and present != predict_present(inf):
        return True, "present_mismatch"

    # 3) Прошедшее / перфект против ВСЕХ слабых классов.
    past = _norm(forms.get("past"))
    perf = strip_aux(forms.get("perfect"))

    if not past and not perf:
        return True, "no_forms_assume_irregular"

    if past and past not in all_weak_pasts(inf):
        return True, "past_not_weak"
    if perf and perf not in all_weak_perfects(inf):
        return True, "perfect_not_weak"

    return False, "regular_weak"


# ── Опции для упражнения-РАЗЛИЧЕНИЯ формы глагола ─────────────────────────────
# Клетки форм глагола в треке форм. Перфект дрилим как ПРИЧАСТИЕ (без 'har'):
# вспомогательный глагол — постоянный контекст, различаем именно причастие.
VERB_FORM_CELLS = ("present", "past", "perfect")


def verb_form_options(inf, forms, cell, n=3):
    """(correct, [distractors]) для клетки-различения формы глагола.
    Дистракторы: РЕАЛЬНЫЕ соседние формы слова (презенс/претерит/причастие/инфинитив) +
    наивная СЛАБО-регулярная форма (для сильных — «ожидаемая, но неверная»: drikke→*drikket, gå→*gådde);
    добор чужими окончаниями классов. Валидные дублеты (kastet/kasta) и малформы отфильтрованы.

    Важно: у СИЛЬНЫХ глаголов all_weak_* даёт механически-слабые формы (=желанные дистракторы),
    поэтому «допустимым» слабый набор считаем ТОЛЬКО когда глагол реально слабый (regular_* ≠ None)."""
    inf = _norm(inf)
    forms = forms or {}

    if cell == "perfect":
        correct = strip_aux(forms.get("perfect"))
    else:
        correct = _norm(forms.get(cell))
    if not correct:
        return None, []

    # Множество ДОПУСТИМЫХ вариантов правильной формы (не предлагаем как дистрактор).
    if cell == "past":
        allowed = set(all_weak_pasts(inf)) if regular_past(inf) is not None else set()
    elif cell == "perfect":
        allowed = set(all_weak_perfects(inf)) if regular_perfect(inf) is not None else set()
    else:  # present — валиден только предсказанный презенс
        allowed = {predict_present(inf)}
    allowed.add(correct)

    pres = _norm(forms.get("present"))
    past = _norm(forms.get("past"))
    part = strip_aux(forms.get("perfect"))
    sib = {"present": pres, "past": past, "perfect": part}

    # наивная слабо-регулярная форма для этой клетки
    _, w_past, w_perf = predict_weak_class(inf)
    naive = {"present": inf + "r",
             "past": w_past,
             "perfect": strip_aux(w_perf) if w_perf else None}[cell]

    # соседние реальные формы (кроме текущей клетки) + наивная + инфинитив-путаница
    primary = [v for c, v in sib.items() if c != cell and v]
    if naive:
        primary.append(_norm(naive))
    primary.append(inf)

    # добор чужими окончаниями классов (чтобы набрать n)
    stem = _weak_stem(inf)
    simp = _degeminate_tail(stem)
    if cell == "present":
        fallback = [inf + "er", stem + "er"]
    elif cell == "past":
        fallback = [stem + "et", stem + "te", stem + "de", simp + "te", simp + "de", inf + "dde"]
    else:  # perfect
        fallback = [stem + "et", stem + "t", stem + "d", simp + "t", simp + "d", inf + "dd"]

    seen, out = set(), []
    for f in primary + fallback:
        if not f or f in allowed or f in seen or not _plausible(f):
            continue
        seen.add(f)
        out.append(f)
        if len(out) >= n:
            break
    return correct, out


# ── Дистракторы для глагола ──────────────────────────────────────────────────

def distractors_verb(inf, forms):
    """Правдоподобные неверные формы глагола.

    Для сильного: слепое слабое правило (gå→*gådde, drikke→*drikket).
    Для слабого: претерит чужого слабого класса (spiste→*spiset/*spisde).
    """
    inf = _norm(inf)
    res = {"present": [], "past": [], "perfect": []}

    correct_present = _norm(forms.get("present"))
    correct_past = _norm(forms.get("past"))
    correct_perf = strip_aux(forms.get("perfect"))

    # present: наивное +r там, где презенс нерегулярен.
    naive_present = inf + "r"
    if naive_present != correct_present:
        res["present"].append(naive_present)
    # для депонентов на -s: *+r всё равно полезный дистрактор (trives→*trivesr)

    # past: все слабые претериты минус правильный.
    res["past"] = sorted(f for f in all_weak_pasts(inf) if f and f != correct_past)
    # частая ошибка: спутать претерит с причастием
    if correct_perf and correct_perf != correct_past and correct_perf not in res["past"]:
        res["past"].append(correct_perf)

    # perfect: все слабые причастия минус правильный (+ '*har drakk' = претерит).
    bad_parts = [p for p in all_weak_perfects(inf) if p and p != correct_perf]
    res["perfect"] = sorted("har " + p for p in bad_parts)
    if correct_past and correct_past != correct_perf:
        cand = "har " + correct_past
        if cand not in res["perfect"]:
            res["perfect"].append(cand)  # *har drakk

    return res
