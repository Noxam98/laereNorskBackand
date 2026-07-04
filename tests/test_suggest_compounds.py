"""Разблокировка составных слов по выученным основам (db suggest_compounds + индекс).

Проверяем связку IO + чистой session.compounds: композит открывается, когда ОБЕ основы
mastered; иначе — нет; повторно не дублируется; счётчик для «Сегодня».
"""
import json

from db.core import _conn, _release
from db import compound_index
from db.learning_suggest import suggest_compounds, unlocked_compounds_count
from tests.conftest import seed_user


async def _seed(dbc, did, uid, no, *, mastered=False, in_dict=True, freq=3.0):
    cur = await dbc.execute(
        "INSERT INTO word_pool (norwegian, data, pos, level, freq, created_at) "
        "VALUES (?,?,?,?,?,datetime('now'))",
        (no, json.dumps({"translate": {"ru": [no]}, "part_of_speech": "noun"}), "noun", "A1", freq))
    pid = cur.lastrowid
    if in_dict:
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
    if mastered:
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, mastered, correct, created_at) "
            "VALUES (?,?,1,4,datetime('now'))", (uid, pid))
    return pid


async def _setup(both_mastered=True):
    uid, did = await seed_user("cmp")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "kjøle", mastered=True)
        await _seed(dbc, did, uid, "skap", mastered=both_mastered)
        cpid = await _seed(dbc, did, uid, "kjøleskap", in_dict=False, freq=4.0)  # композита у юзера НЕТ
        await dbc.commit()
    finally:
        await _release(dbc)
    compound_index._CACHE["index"] = None                       # сбрасываем TTL-кеш между тестами
    await compound_index.set_pool_compounds([(cpid, "kjøleskap", "kjøle", "skap")])
    return uid, cpid


async def test_unlock_when_both_parts_mastered(fresh_db):
    uid, _cpid = await _setup(both_mastered=True)
    assert (await suggest_compounds(uid))["added"] == 1        # обе основы выучены → открыт
    assert (await suggest_compounds(uid))["added"] == 0        # уже в словаре → не дублируем


async def test_not_unlocked_if_one_part_missing(fresh_db):
    uid, _cpid = await _setup(both_mastered=False)             # skap не mastered
    assert (await suggest_compounds(uid))["added"] == 0
    assert await unlocked_compounds_count(uid) == 0


async def test_unlocked_count_before_learning(fresh_db):
    uid, _cpid = await _setup(both_mastered=True)
    assert await unlocked_compounds_count(uid) == 1           # открыт, но ещё не в словаре
    await suggest_compounds(uid)
    assert await unlocked_compounds_count(uid) == 0          # добавлен → больше не «ожидает»


async def test_compounds_unlocked_by_root_for_celebration(fresh_db):
    """Празднование: выучив основу, считаем НОВО открытые композиты (вторая часть уже выучена)."""
    from db.compound_index import compounds_unlocked_by
    uid, cpid = await _setup(both_mastered=True)   # kjøle+skap mastered, kjøleskap не в словаре
    dbc = await _conn()
    try:
        assert await compounds_unlocked_by(dbc, uid, "skap") == 1    # выучил skap → открыл kjøleskap
        assert await compounds_unlocked_by(dbc, uid, "kjøle") == 1
        assert await compounds_unlocked_by(dbc, uid, "annet") == 0   # не часть композита
        # композит уже в словаре юзера → не «новый»
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) "
                          "SELECT id, ?, datetime('now') FROM dictionaries WHERE user_id=? LIMIT 1", (cpid, uid))
        await dbc.commit()
        assert await compounds_unlocked_by(dbc, uid, "skap") == 0
    finally:
        await _release(dbc)
