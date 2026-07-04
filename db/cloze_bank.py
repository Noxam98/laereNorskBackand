"""Статический банк cloze для служебных слов A1-A2 (data/cloze-bank.json).

Заменяет пер-юзерную LLM-генерацию (learning_cloze._generate_cloze): предложения
и дистракторы выверены офлайн (рой Sonnet + адверсарная валидация), а не строятся на
лету из выученных слов юзера. Почему банк, а не генерация:
  - дистракторы — служебные слова ДРУГОГО типа связи, грамматически НЕ подходящие в
    данный пропуск (не синонимы-соседи по эмбеддингу, из-за которых «samt» давал 4
    правки-синонима — исходный баг);
  - предложения из фиксированного ядра A1-A2, всегда осмысленны, не зависят от того,
    что юзер уже выучил (и не жрут квоту Gemini на каждого).

Файл — как data/levels-v1.json: грузится один раз при импорте. Отсутствует/битый →
пустой банк (has()==False везде) → служебные идут упрощённой рампой FUNC_CHOICE
(текущее прод-поведение при CLOZE_ENABLED=0), ничего не ломается.

Публичное: has(no, pos) — есть ли выверенный cloze; items_for(no, pos) — сырые items
[{blank, answer, distractors}] (сборку options делает get_cloze_map). Ключ —
(norwegian_lower, canonical_pos): у омонимов (for-предлог/for-союз, da-наречие/da-союз)
разные записи пула и разный cloze; поэтому pos участвует в ключе. Слова с единственной
записью (в т.ч. «å») находятся по norwegian без совпадения pos."""
import json
import os

try:
    from pos import normalize_pos
except Exception:                       # pos-модуль недоступен (изоляция тестов) — деградируем мягко
    def normalize_pos(p):
        return (p or "").strip().lower()

_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cloze-bank.json")

_BY_NO = {}     # norwegian_lower -> [ {pos, lvl, items:[{blank,answer,distractors}]} , ... ]
_LOADED = False


def _load():
    """Лениво прочитать банк один раз. Битый/отсутствующий файл → пустой банк (без исключений)."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        with open(_PATH, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return
    for entry in (raw.get("words") or []):
        no = (entry.get("no") or "").strip().lower()
        items = [it for it in (entry.get("items") or [])
                 if it.get("blank") and it.get("answer") and it.get("distractors")]
        if not no or not items:
            continue
        _BY_NO.setdefault(no, []).append({
            "pos": normalize_pos(entry.get("pos")),
            "lvl": entry.get("lvl"),
            "items": items,
        })


def _lookup(no, pos):
    """Запись банка для (norwegian, pos) или None. Одна запись на слово → берём её без
    сверки pos (покрывает «å» и все неомонимы); несколько → сверяем canonical pos."""
    _load()
    cands = _BY_NO.get((no or "").strip().lower())
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    want = normalize_pos(pos)
    for c in cands:
        if c["pos"] == want:
            return c
    return cands[0]                      # pos не совпал (данные разошлись) — не оставляем без cloze


def has(no, pos):
    """Есть ли в банке выверенный cloze для этого служебного слова (гейтит рампу FUNC_CLOZE)."""
    return _lookup(no, pos) is not None


def items_for(no, pos):
    """Сырые cloze-items [{blank, answer, distractors}] или None. options (перемешанные
    answer+distractors) собирает get_cloze_map — банк хранит только данные."""
    e = _lookup(no, pos)
    return e["items"] if e else None
