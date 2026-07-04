"""Формы местоимений/притяжательных в карточке (get_pool_meta).

Сущ./глаг./прил. хранят формы в word_pool.forms (ordbank), а местоимения/притяж. — нет:
get_pool_meta достаёт их из курируемой PRONOUN_PARADIGM, чтобы карточка показывала формы
для ВСЕХ частей речи, у которых они есть в системе.
"""
import json

from db.core import _conn, _release
from db.pool import get_pool_meta


async def _seed(no, pos):
    dbc = await _conn()
    try:
        await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            (no, json.dumps({"translate": {"ru": ["x"]}, "part_of_speech": pos}), pos, "A1"))
        await dbc.commit()
    finally:
        await _release(dbc)


async def test_personal_pronoun_object_form(fresh_db):
    await _seed("jeg", "pronoun")
    meta = await get_pool_meta("jeg")
    assert meta["forms"] == {"pos": "pronoun", "obj": "meg"}    # подлежащее jeg / дополнение meg


async def test_possessive_neuter_plural_forms(fresh_db):
    await _seed("min", "determiner")
    meta = await get_pool_meta("min")
    assert meta["forms"]["pos"] == "pronoun"
    assert meta["forms"]["neuter"] == "mitt" and meta["forms"]["plural"] == "mine"


async def test_non_paradigm_word_keeps_none(fresh_db):
    await _seed("katt", "noun")   # не в парадигме, форм в колонке нет, банка в тестах нет → None
    meta = await get_pool_meta("katt")
    assert meta["forms"] is None
