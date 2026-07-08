"""Фиксы embcache/дистракторов (чистая логика, без сети/venv/numpy).

Покрываем:
  #3 — фолбэк добивает out до n, когда соседей нет/мало (иначе пустая choice-карточка);
  #4 — дистрактор-ОМОНИМ (то же норв. написание, другой смысл) отсекается в no2int,
       иначе его перевод показан вариантом, а он тоже верен к показанному норв. слову.
"""
from session import distractors as dx


def _cand(no, ru):
    """Кандидат-слово в форме, которую приносит оркестратор (как words[id] / fallback[pid])."""
    return {"norwegian": no, "data": {"translate": {"no": [no], "ru": list(ru)}}}


def _item(pid=1, no="hus", ru=("дом",), direction="int2no"):
    return {"pool_id": pid, "no": no, "mode": "choice", "direction": direction,
            "translate": {"no": [no], "ru": list(ru)}, "own_options": False}


# ---- #3: фолбэк ----

def test_fallback_fills_when_no_neighbors():
    """Соседей нет вовсе (холодный кеш / слово без эмбеддинга) → добираем до n из фолбэка."""
    fb = {1: [_cand("bok", ["книга"]), _cand("bil", ["машина"]), _cand("vei", ["дорога"])]}
    patch = dx.options_patch([_item()], neighbors={}, words={}, lang="ru", fallback=fb)
    assert [o["w"] for o in patch[1]["options"]] == ["bok", "bil", "vei"]
    assert patch[1]["distractors"] == ["bok", "bil", "vei"]


def test_fallback_tops_up_partial_neighbors():
    """Соседей меньше n → фолбэк ДОБИРАЕТ хвост (приоритет у соседей)."""
    neighbors = {1: [10]}
    words = {10: _cand("bok", ["книга"])}
    fb = {1: [_cand("bil", ["машина"]), _cand("vei", ["дорога"])]}
    patch = dx.options_patch([_item()], neighbors=neighbors, words=words, lang="ru", n=3, fallback=fb)
    assert [o["w"] for o in patch[1]["options"]] == ["bok", "bil", "vei"]


def test_fallback_unused_when_neighbors_enough():
    """Соседей хватает — фолбэк не подмешивается (не нужен)."""
    neighbors = {1: [10, 11, 12]}
    words = {10: _cand("bok", ["книга"]), 11: _cand("bil", ["машина"]), 12: _cand("vei", ["дорога"])}
    fb = {1: [_cand("sjø", ["море"])]}
    patch = dx.options_patch([_item()], neighbors=neighbors, words=words, lang="ru", n=3, fallback=fb)
    assert [o["w"] for o in patch[1]["options"]] == ["bok", "bil", "vei"]


def test_fallback_applies_same_filters():
    """К фолбэку те же фильтры: синоним по смыслу и омоним по написанию не проходят."""
    fb = {1: [
        _cand("hjem", ["дом", "жильё"]),   # синоним цели (пересечение смысла) — отсечь
        _cand("hus", ["корпус"]),          # то же норв. написание, что цель — омоним, отсечь
        _cand("bil", ["машина"]),          # годный
    ]}
    patch = dx.options_patch([_item()], neighbors={}, words={}, lang="ru", fallback=fb)
    assert [o["w"] for o in patch[1]["options"]] == ["bil"]


def test_no_fallback_stays_empty():
    """Без фолбэка и без соседей поведение прежнее — пустые options (не исключение)."""
    patch = dx.options_patch([_item()], neighbors={}, words={}, lang="ru")
    assert patch[1] == {"options": [], "distractors": []}


# ---- #4: омоним по норвежскому написанию в no2int ----

def test_homograph_by_norwegian_excluded_no2int():
    """no2int: показываем норв. «ro», ждём перевод. Дистрактор «ro» другого смысла (грести)
    тоже верен к показанному слову → его перевод НЕ должен попасть в варианты."""
    item = _item(no="ro", ru=("покой",), direction="no2int")
    neighbors = {1: [20, 21]}
    words = {
        20: {"norwegian": "ro", "data": {"translate": {"ru": ["грести"]}}},   # омоним — то же написание
        21: {"norwegian": "hus", "data": {"translate": {"ru": ["дом"]}}},
    }
    patch = dx.options_patch([item], neighbors=neighbors, words=words, lang="ru")
    ws = [o["w"] for o in patch[1]["options"]]
    assert "грести" not in ws
    assert ws == ["дом"]


def test_homograph_via_translate_no_forms_excluded():
    """Омоним ловим и по формам translate.no цели, не только по e['no']."""
    item = {"pool_id": 1, "no": "vær", "mode": "choice", "direction": "no2int",
            "translate": {"no": ["vær", "være"], "ru": ["погода"]}, "own_options": False}
    neighbors = {1: [30, 31]}
    words = {
        30: {"norwegian": "være", "data": {"translate": {"ru": ["быть"]}}},   # совпало с translate.no
        31: {"norwegian": "sol", "data": {"translate": {"ru": ["солнце"]}}},
    }
    patch = dx.options_patch([item], neighbors=neighbors, words=words, lang="ru")
    assert [o["w"] for o in patch[1]["options"]] == ["солнце"]
