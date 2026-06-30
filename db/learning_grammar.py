"""Грамм-overlay (тир ★): клетки проверки форм слова ПОВЕРХ base-рампы. Вынесено из learning.py.

Отдельный слой — на «выучено»/CEFR не влияет (см. apply_result). Форму проверяем ТОЛЬКО когда она
нерегулярна (факт ≠ предсказанию морфологии); регулярную пользователь выводит правилом. Часть речи:
сущ. (род + неопр.мн.), глаг. (претерит/перфект/презенс), прил. (ср.род/степени/мн.),
местоим./притяж. (падеж + согласование, курируемая парадигма). Публичное ядру: _grammar_cells,
_grammar_element (реэкспортируются обратно в learning, см. конец learning.py)."""
import json
import unicodedata

from pos import normalize_pos
from morphology import (
    is_irregular_noun, regular_indef_pl,                                          # сущ.
    predict_present, all_weak_pasts, all_weak_perfects, strip_aux,                # глаг.
    regular_neuter, regular_plural, regular_comparative, regular_superlative, is_irregular_adj,  # прил.
)
from .learning import _GENDERS, _INPUT_FORM_CELLS, PRONOUN_PARADIGM


def _nfc_low(s):
    """Форму к NFC + lower для сверки: формы из БД бывают в NFD → å/ø/æ иначе не совпали бы."""
    return unicodedata.normalize("NFC", (s or "").strip()).lower()


def _noun_grammar_cells(no, f):
    """Сущ.: choice_gender (есть род) + input_indefpl (мн.ч. нерегулярно). input — при валидном роде
    (предиктор опирается на род) и NFC-сверке (иначе ложный дрилл / цель, которую не ввести)."""
    out = []
    gender = (f.get("gender") or "").strip()
    if gender in _GENDERS:
        out.append("choice_gender")
    indef_pl = (f.get("indef_pl") or "").strip()
    if indef_pl and gender in _GENDERS:
        reg = regular_indef_pl(no, gender)
        if reg:
            if _nfc_low(indef_pl) != _nfc_low(reg):           # факт ≠ правило → форму надо знать
                out.append("input_indefpl")
        elif is_irregular_noun(no, gender, f)[0]:             # правило не предсказало → детектор
            out.append("input_indefpl")
    return out


def _verb_grammar_cells(inf, f):
    """Глаг.: ввод формы, если она НЕ выводится слабым правилом (членство в слабых классах) — т.е.
    сильные/супплетивные (gå→gikk, skrive→skrev). Презенс — лишь вне таблиц predict_present (редко).
    Сравниваем по членству, а НЕ по «≠ канону»: слабые двухклассовые (lage→laget) иначе дали бы
    ложный дрилл."""
    out = []
    past = (f.get("past") or "").strip()
    if past and _nfc_low(past) not in {_nfc_low(x) for x in all_weak_pasts(inf)}:
        out.append("input_past")
    perf = (f.get("perfect") or "").strip()
    if perf:
        part = _nfc_low(strip_aux(perf))                      # причастие без 'har '/'er '
        if part and part not in {_nfc_low(x) for x in all_weak_perfects(inf)}:
            out.append("input_perfect")
    present = (f.get("present") or "").strip()
    pred = predict_present(inf)
    if present and pred and _nfc_low(present) != _nfc_low(pred):
        out.append("input_present")
    return out


def _adj_grammar_cells(lemma, f):
    """Прил.: ввод формы, если факт ≠ предсказанию правила, либо супплетив (предиктор вернул None и
    детектор подтвердил нерегулярность: god→bedre→best, liten→små). Перифразы (mer/mest …) — мимо."""
    out = []
    irr = {}
    def _irregular():
        if "v" not in irr:
            irr["v"] = is_irregular_adj(lemma, f)
        return irr["v"][0]

    def _add(field, cell, predictor):
        val = (f.get(field) or "").strip()
        if not val:
            return
        pred = predictor(lemma)
        if pred:
            if _nfc_low(val) != _nfc_low(pred):
                out.append(cell)
        elif _irregular():
            out.append(cell)

    _add("neuter", "input_neuter", regular_neuter)
    if " " not in (f.get("comparative") or ""):        # не перифраз 'mer …'
        _add("comparative", "input_comparative", regular_comparative)
    if " " not in (f.get("superlative") or ""):        # не 'mest …'
        _add("superlative", "input_superlative", regular_superlative)
    plural = (f.get("plural") or "").strip()
    if plural:
        pred = regular_plural(lemma)
        if pred:
            if _nfc_low(plural) != _nfc_low(pred):
                out.append("input_pluraladj")
        elif _irregular() and _nfc_low(plural) != _nfc_low(lemma):   # liten→små; не лемма-дублет
            out.append("input_pluraladj")
    return out


def _pronoun_grammar_cells(no, f):
    """Местоимение/притяж. (курируемая парадигма в forms): объектный падеж (jeg→meg, если ≠ субъекту)
    + согласование притяжательных (ср.род mitt, мн. mine). Закрытый класс — без морфологии-правил."""
    out = []
    obj = (f.get("obj") or "").strip()
    if obj and _nfc_low(obj) != _nfc_low(no):
        out.append("input_objcase")
    if (f.get("neuter") or "").strip():
        out.append("input_possneut")
    if (f.get("plural") or "").strip():
        out.append("input_posspl")
    return out


def _grammar_cells(norwegian, data, forms):
    """Грамм-клетки (overlay-тир ★) слова — по части речи + наличию форм. НЕ влияют на base-рампу,
    «выучено» и CEFR (отдельный слой, см. apply_result). Проверяем форму ТОЛЬКО когда она нерегулярна
    (факт ≠ предсказанию морфологии) — регулярную пользователь выводит правилом. Сущ.: choice_gender +
    input_indefpl. Глаг.: input_past/perfect/present. Прил.: input_neuter/comparative/superlative/
    pluraladj."""
    try:
        d = json.loads(data) if isinstance(data, str) else (data or {})
    except Exception:
        d = {}
    pos = normalize_pos(d.get("part_of_speech"))
    if pos not in ("noun", "verb", "adjective", "pronoun", "determiner"):
        return []
    try:
        f = json.loads(forms) if isinstance(forms, str) else (forms or {})
    except Exception:
        f = {}
    no = norwegian or ""
    if not f and pos in ("pronoun", "determiner"):
        f = PRONOUN_PARADIGM.get(no.strip().lower())   # курируемая парадигма (форм нет в БД / forms_loop)
    if not f:
        return []
    if pos == "noun":
        return _noun_grammar_cells(no, f)
    if pos == "verb":
        return _verb_grammar_cells(no, f)
    if pos == "adjective":
        return _adj_grammar_cells(no, f)
    return _pronoun_grammar_cells(no, f)   # pronoun / determiner — курируемая парадигма (seed_pronoun_forms)


def _grammar_element(row, cell, forms, data):
    """Сессионный элемент грамм-overlay по клетке (параметризованный контракт target/prompt/…).
    grammar:True — фронт рендерит FormPrompt, а _attach_choice_options НЕ трогает его варианты."""
    no = row["norwegian"]
    pos = normalize_pos((data or {}).get("part_of_speech")) or "noun"
    base = {
        "pool_id": row["pool_id"], "no": no, "translate": (data or {}).get("translate", {}),
        "part_of_speech": pos, "forms": forms,
        "step": cell, "grammar": True, "repeat": (row.get("mastered") == 1),
    }
    if cell == "choice_gender":
        g = (forms.get("gender") or "").strip()
        return {**base, "mode": "choice", "direction": "gender",
                "target": {"field": "gender", "value": g},
                "prompt": {"kind": "lemma+formLabel", "formLabel": "gender", "lemma": no},
                "options": [{"w": a, "alt": None} for a in _GENDERS],   # порядок перемешает фронт
                "distractors": [a for a in _GENDERS if a != g]}
    if cell in _INPUT_FORM_CELLS:   # все input-формы (сущ./глаг./прил./местоим.): ввод формы
        field, label = _INPUT_FORM_CELLS[cell]
        return {**base, "mode": "input", "direction": cell.split("_", 1)[1],
                "target": {"field": field, "value": (forms.get(field) or "").strip()},
                "prompt": {"kind": "lemma+formLabel", "formLabel": label, "lemma": no},
                "scoring": {"typoForgive": False}}
    return None
