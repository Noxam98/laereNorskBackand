"""Трек ФОРМ (bokmål): рампа изучения словоформ поверх выученного слова — ЧИСТАЯ логика (без БД).

Отличие от грамм-overlay (learning_grammar): трек гоняет КАЖДУЮ форму слова (в т.ч. регулярную),
имеет собственную рампу карточка→выбор→ввод и СВОЁ расписание (SM-2 на клетку — schedule_form;
хранилище form_srs и apply подключаются на слое БД отдельным шагом). Дистракторы «выбора» —
динамическая подмена окончаний из morphology.*_form_options: различаем нужную форму среди
правдоподобных ошибок (gå→*gådde, grønn→*grønnt), не «угадайка» по чужим словам.

Здесь только КАКОЕ задание показать и КАК сдвинуть расписание — вызывается и планировщиком, и сессией."""
import json
import unicodedata

from pos import normalize_pos
from morphology import (
    strip_aux,
    noun_form_options, verb_form_options, adj_form_options,
    NOUN_FORM_CELLS, VERB_FORM_CELLS, ADJ_FORM_CELLS,
)

# Рампа одной формы: показать карточку → выбрать верную среди подменённых окончаний → набрать самому.
FORM_STAGES = ("card", "choose", "produce")

# Части речи с морфологией форм (местоим./детерминативы — курируемая парадигма, не сюда).
FORM_CELLS_BY_POS = {
    "noun": NOUN_FORM_CELLS,
    "verb": VERB_FORM_CELLS,
    "adjective": ADJ_FORM_CELLS,
}

# Клетка формы → (поле в forms, метка-подпись для фронта i18n). gender — особая (выбор артикля).
_FORM_FIELD = {
    "gender":      ("gender",      "gender"),
    "indef_pl":    ("indef_pl",    "indef_pl"),
    "def_sg":      ("def_sg",      "def_sg"),
    "def_pl":      ("def_pl",      "def_pl"),
    "present":     ("present",     "present"),
    "past":        ("past",        "past"),
    "perfect":     ("perfect",     "perfect"),
    "neuter":      ("neuter",      "neuter"),
    "plural":      ("plural",      "plural_adj"),
    "comparative": ("comparative", "comparative"),
    "superlative": ("superlative", "superlative"),
}

# SM-2 (как база, см. learning.apply_result) — держим тут, чтобы модуль был самодостаточным.
_EASE_START, _EASE_MAX, _EASE_MIN = 2.5, 3.0, 1.3
_INTERVAL_CAP = 365
_FAST_SEC = 6.0


def _pos_of(data):
    try:
        d = json.loads(data) if isinstance(data, str) else (data or {})
    except Exception:
        d = {}
    return normalize_pos(d.get("part_of_speech"))


def parse_forms(forms):
    try:
        return json.loads(forms) if isinstance(forms, str) else (forms or {})
    except Exception:
        return {}


def cell_value(pos, forms, cell):
    """Значение клетки формы. Перфект глагола дрилим как ПРИЧАСТИЕ (без 'har' — вспом. постоянен)."""
    if cell == "gender":
        return (forms.get("gender") or "").strip()
    field = _FORM_FIELD.get(cell, (cell, cell))[0]
    val = (forms.get(field) or "").strip()
    if pos == "verb" and cell == "perfect" and val:
        return strip_aux(val)
    return val


def form_cells_for(pos, forms):
    """Клетки форм, реально доступные слову: часть речи из FORM_CELLS_BY_POS + форма присутствует и
    не пуста (несклоняемое/отсутствующее — пропускаем; перифразы 'mer/mest …' — тоже, это не одна форма)."""
    cells = FORM_CELLS_BY_POS.get(pos)
    if not cells:
        return []
    out = []
    for c in cells:
        v = cell_value(pos, forms, c)
        if v and " " not in v:          # 'mer praktisk' и т.п. — не одна словоформа, мимо
            out.append(c)
    return out


def form_options(pos, no, forms, cell, n=3):
    """(correct, [distractors]) для клетки формы — диспетч в морфологию по части речи."""
    if pos == "noun":
        return noun_form_options(no, (forms.get("gender") or "").strip(), forms, cell, n)
    if pos == "verb":
        return verb_form_options(no, forms, cell, n)
    if pos == "adjective":
        return adj_form_options(no, forms, cell, n)
    return (None, [])


def form_element(row, forms, data, cell, stage):
    """Сессионный элемент трека форм для клетки+ступени. Совместим с контрактом грамм-элемента
    (mode/direction/target/prompt) + флаг form_track:True (фронт роутит ответ в трек, не в base-рампу).
    card — пассивный показ формы; choose — выбор среди подменённых окончаний; produce — ввод.
    Возвращает None, если для выбора не удалось собрать варианты (нет correct)."""
    d = data if isinstance(data, dict) else parse_forms(data)
    pos = normalize_pos(d.get("part_of_speech")) or "noun"
    no = row["norwegian"]
    field, label = _FORM_FIELD.get(cell, (cell, cell))
    value = cell_value(pos, forms, cell)
    base = {
        "pool_id": row["pool_id"], "no": no, "translate": d.get("translate", {}),
        "part_of_speech": pos, "forms": forms, "form_track": True,
        "step": cell, "stage": stage, "repeat": (row.get("mastered") == 1),
        "prompt": {"kind": "lemma+formLabel", "formLabel": label, "lemma": no},
    }
    if stage == "card":                       # показать форму (пассив, как карточка перевода)
        return {**base, "mode": "study", "direction": cell,
                "target": {"field": field, "value": value}, "reveal": value}
    if stage == "choose":                     # выбрать верную среди динамически подменённых окончаний
        correct, dis = form_options(pos, no, forms, cell)
        if not correct:
            return None
        return {**base, "mode": "choice", "direction": cell,
                "target": {"field": field, "value": correct},
                "options": [{"w": w, "alt": None} for w in [correct] + dis],
                "distractors": dis}
    # produce — набрать форму самому
    return {**base, "mode": "input", "direction": cell,
            "target": {"field": field, "value": value}, "scoring": {"typoForgive": False}}


def schedule_form(stage, ease, interval_days, correct, elapsed=None):
    """Чистый шаг планировщика клетки формы (SM-2, как база). Возвращает
    (next_stage, ease, interval_days, due_days). due_days=0 → повтор в ЭТОЙ же сессии.

    Рампа: card (пассив) → choose → produce. Верно двигает ступень; на produce верно → планируем
    повтор (interval*ease). Ошибка → ease вниз, шаг назад, скорый повтор."""
    ease = ease or _EASE_START
    interval_days = interval_days or 0
    idx = FORM_STAGES.index(stage) if stage in FORM_STAGES else 0

    if stage == "card":                        # пассивный показ — сразу к выбору, расписание не трогаем
        return ("choose", ease, interval_days, 0)

    if correct:
        fast = elapsed is not None and elapsed <= _FAST_SEC
        ease = min(_EASE_MAX, ease + (0.08 if fast else 0.04))
        if idx < len(FORM_STAGES) - 1:         # choose→produce: ещё в этой сессии
            return (FORM_STAGES[idx + 1], ease, interval_days, 0)
        interval = 1 if interval_days < 1 else min(_INTERVAL_CAP, round(interval_days * ease))
        if fast and interval < 1:
            interval = 2
        return ("produce", ease, interval, interval)   # клетка отработана → повтор через interval дней

    ease = max(_EASE_MIN, ease - 0.2)          # ошибка: ease вниз, шаг назад, повтор в этой сессии
    return (FORM_STAGES[max(0, idx - 1)], ease, 1, 0)
