"""ЭТАП 7: дистракторы choice-элементов как чистый патч (session/distractors.py).

Замки: вход не мутируется; own_options-элементы (грамм-тир) не перезаписываются;
синоним по смыслу не попадает в варианты; холодный embcache — лог, а не тишина.
"""
import copy

from session import distractors as dx


def _item(pid=1, no="hus", ru=("дом",), direction="int2no"):
    return {"pool_id": pid, "no": no, "mode": "choice", "direction": direction,
            "translate": {"no": [no], "ru": list(ru)}, "own_options": False}


NBR = {1: [10, 11, 12, 13]}
WORDS = {
    10: {"norwegian": "bok", "data": {"translate": {"ru": ["книга"]}}},
    11: {"norwegian": "hjem", "data": {"translate": {"ru": ["дом", "жильё"]}}},   # синоним цели
    12: {"norwegian": "bil", "data": {"translate": {"ru": ["машина"]}}},
    13: {"norwegian": "vei", "data": {"translate": {"ru": ["дорога"]}}},
}


def test_patch_int2no_skips_synonym_and_caps_n():
    item = _item()
    patch = dx.options_patch([item], neighbors=NBR, words=WORDS, lang="ru")
    got = patch[1]
    assert [o["w"] for o in got["options"]] == ["bok", "bil", "vei"]   # hjem-синоним исключён
    assert got["distractors"] == ["bok", "bil", "vei"]


def test_patch_no2int_uses_translations():
    item = _item(direction="no2int")
    patch = dx.options_patch([item], neighbors=NBR, words=WORDS, lang="ru")
    assert [o["w"] for o in patch[1]["options"]] == ["книга", "машина", "дорога"]


def test_input_not_mutated():
    item = _item()
    before = copy.deepcopy(item)
    dx.options_patch([item], neighbors=NBR, words=WORDS, lang="ru")
    assert item == before


def test_apply_patch_never_overwrites_own_options():
    grammar_el = {"pool_id": 1, "mode": "choice", "own_options": True,
                  "options": [{"w": "et", "alt": None}], "distractors": ["et"]}
    legacy_el = {"pool_id": 3, "mode": "choice", "grammar": True}   # до-Этап-6, без own_options
    content_el = _item(2)
    session = [grammar_el, legacy_el, content_el]
    assert dx.choice_targets(session) == [content_el]
    patch = {1: {"options": [{"w": "X", "alt": None}], "distractors": ["X"]},
             2: {"options": [{"w": "Y", "alt": None}], "distractors": ["Y"]},
             3: {"options": [{"w": "Z", "alt": None}], "distractors": ["Z"]}}
    dx.apply_patch(session, patch)
    assert grammar_el["options"] == [{"w": "et", "alt": None}]   # свои варианты целы
    assert "options" not in legacy_el                            # легаси-grammar тоже не тронут
    assert content_el["options"] == [{"w": "Y", "alt": None}]    # контентный получил патч


def test_cold_cache_empty_options():
    """Соседей нет (холодный кеш) → патч с пустыми options, не исключение."""
    patch = dx.options_patch([_item()], neighbors={}, words={}, lang="ru")
    assert patch[1] == {"options": [], "distractors": []}


async def test_cold_embcache_logs_degradation(monkeypatch, caplog):
    """Оркестратор: пустой embcache — WARNING в лог (фронт без вариантов ≠ норма)."""
    import embcache
    from db import learning

    async def _empty(pids, k):
        return {}

    monkeypatch.setattr(embcache, "candidates_for", _empty)
    session = [_item()]
    with caplog.at_level("WARNING"):
        await learning._attach_choice_options(session, "ru")
    assert any("embcache" in r.message for r in caplog.records)
    assert session[0]["options"] == []
