"""ЭТАП 6: единая форма элемента сессии (session/shape.py).

Замки: замороженные BASE_KEYS/EXTRA_KEYS (осознанная правка = правка теста),
конструктор отвергает неизвестные ключи и choice без своих options при
own_options=True, все билдеры (дневная сессия / слух / живой добор / грамм-тир)
дают один базовый набор ключей — дрейф формы между /learning/session и
/learning/next-cards закрыт.
"""
import json

import pytest

from session.shape import make_element, BASE_KEYS, EXTRA_KEYS
from db.core import _conn, _release
from tests.conftest import seed_user


def _mk(**over):
    kw = dict(pool_id=1, no="hus", translate={"ru": ["дом"]}, part_of_speech="noun",
              forms=None, mode="study", direction=None, step="card", repeat=False)
    kw.update(over)
    return make_element(**kw)


def test_base_keys_always_present_and_frozen():
    el = _mk()
    assert tuple(el) == BASE_KEYS      # состав и порядок базы — контракт
    assert el["grammar"] is False and el["own_options"] is False
    # правка формы элемента = ОСОЗНАННАЯ правка этих литералов (вместе с фронтом)
    assert BASE_KEYS == ("pool_id", "no", "translate", "part_of_speech", "forms",
                         "mode", "direction", "step", "repeat", "grammar", "own_options")
    assert EXTRA_KEYS == frozenset((
        "gloss", "example", "listen", "cloze", "options", "distractors",
        "form_track", "stage", "prompt", "target", "reveal", "scoring", "compound"))


def test_unknown_extra_key_rejected():
    with pytest.raises(ValueError):
        _mk(surprise=1)


def test_own_options_choice_requires_options():
    with pytest.raises(ValueError):
        _mk(mode="choice", own_options=True)
    el = _mk(mode="choice", own_options=True, options=[{"w": "en", "alt": None}])
    assert el["options"]


def test_grammar_builders_go_through_make_element():
    """form_element/_grammar_element: база на месте, grammar+own_options проставлены."""
    from db.learning_forms import form_element
    from db.learning_grammar import _grammar_element
    forms = {"gender": "et", "def_sg": "huset", "indef_pl": "hus", "def_pl": "husene"}
    row = {"pool_id": 7, "norwegian": "hus", "mastered": 1}
    data = {"part_of_speech": "noun", "translate": {"ru": ["дом"]}}
    els = [form_element(row, forms, data, "def_sg", "card"),
           form_element(row, forms, data, "gender", "choose"),
           form_element(row, forms, data, "def_sg", "produce"),
           _grammar_element(row, "choice_gender", forms, data)]
    for el in els:
        assert el is not None
        assert set(BASE_KEYS) <= set(el)
        assert el["grammar"] is True and el["own_options"] is True
        assert set(el) <= set(BASE_KEYS) | EXTRA_KEYS
    assert els[1]["mode"] == "choice" and els[1]["options"]   # свои варианты при choice


async def _seed(dbc, did, uid, no, *, modes=None):
    cur = await dbc.execute(
        "INSERT INTO word_pool (norwegian, data, pos, level, freq, created_at) "
        "VALUES (?,?,?,?,?,datetime('now'))",
        (no, json.dumps({"translate": {"no": [no], "ru": ["слово"]}, "part_of_speech": "noun",
                         "gloss": "g", "example": "ex"}), "noun", "A1", 3.0))
    pid = cur.lastrowid
    await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) "
                      "VALUES (?,?,datetime('now'))", (did, pid))
    if modes is not None:
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, strength, reps, correct, incorrect, "
            "interval_days, due_at, modes, created_at) "
            "VALUES (?,?,60,4,4,0,2,datetime('now','+1 day'),?,datetime('now'))",
            (uid, pid, json.dumps(modes)))
    return pid


async def test_same_base_shape_across_all_builders(fresh_db):
    """Одно состояние юзера через 3 БД-билдера: базовые ключи идентичны, лишних нет."""
    from db.learning import build_session, build_listen_session
    from db.learning_suggest import next_new_cards
    uid, did = await seed_user("shape")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "nyord")                    # новое → карточка
        await _seed(dbc, did, uid, "lytteord", modes={          # текст сдан → ждёт слух
            "choice_int2no": "1", "build_int2no": "1", "input_int2no": "1"})
        await dbc.commit()
    finally:
        await _release(dbc)

    day = (await build_session(uid, size=10))["words"]
    listen = (await build_listen_session(uid))["words"]
    cards = (await next_new_cards(uid, n=1))["cards"]
    assert day and listen and cards
    for el in day + listen + cards:
        assert set(BASE_KEYS) <= set(el), el
        assert set(el) <= set(BASE_KEYS) | EXTRA_KEYS, el
    # значения слухового элемента — контракт фронта (плеер + прогресс-бар по стадии)
    li = listen[0]
    assert (li["mode"], li["direction"], li["step"], li["listen"]) == \
        ("choice", "no2int", "choice_no2int", True)
    assert li["gloss"] == "g" and li["example"] == "ex"        # union Этапа 6 (было только в дневной)
    # дневная карточка и живой добор ОДНОГО слова — идентичная форма
    card_day = next(el for el in day if el["step"] == "card")
    assert card_day["no"] == cards[0]["no"] == "nyord"
    assert set(card_day) == set(cards[0])
