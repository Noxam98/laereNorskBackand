"""Скрытый авто-словарь («докинуть») + флаг «в обучении» (studying) на словаре.

Спека: §7 «Связь словаря и Учёбы».
- Учёба берёт слова только из словарей со studying=1.
- «Докинуть»/стартовый набор кладёт слова в СКРЫТЫЙ авто-словарь (hidden=1, studying=1).
- Скрытый авто-словарь не виден в «Мой словарь», но виден Учёбе.
- Скрытый авто-словарь нельзя выключить из обучения.
"""
import json
import pytest

from db import core
from db.core import _conn, _release, _now
from db.dictionaries import (
    add_word_to_dict, get_user_data, get_or_create_hidden_dict,
    set_dictionary_studying, HIDDEN_DICT_NAME,
)
from db.learning import _fetch_user_words


async def _add_pool_word(no, ru="x", level="A1"):
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
            (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _learning_pools(user_id):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        return {r["pool_id"] for r in rows}
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_hidden_dict_get_or_create_is_stable(fresh_db):
    uid, _did = await pytest.seed_user()
    h1 = await get_or_create_hidden_dict(uid)
    h2 = await get_or_create_hidden_dict(uid)
    assert h1 == h2


@pytest.mark.asyncio
async def test_hidden_dict_excluded_from_user_data_but_in_learning(fresh_db):
    uid, did = await pytest.seed_user()
    pid_personal, _ = await pytest.seed_word(did, "hus")
    h = await get_or_create_hidden_dict(uid)
    pid_hidden = await _add_pool_word("katt", "кот")
    await add_word_to_dict(uid, h, pid_hidden)

    data = await get_user_data(uid)
    names = [d["dictName"] for d in data["dictList"]]
    assert HIDDEN_DICT_NAME not in names
    assert all("studying" in d for d in data["dictList"])

    pools = await _learning_pools(uid)
    assert pid_personal in pools
    assert pid_hidden in pools  # скрытый studying=1 → в Учёбе виден


@pytest.mark.asyncio
async def test_studying_off_drops_words_from_learning(fresh_db):
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "bil")
    assert pid in await _learning_pools(uid)

    res = await set_dictionary_studying(uid, did, False)
    assert res["studying"] is False
    assert pid not in await _learning_pools(uid)

    res = await set_dictionary_studying(uid, did, True)
    assert res["studying"] is True
    assert pid in await _learning_pools(uid)


@pytest.mark.asyncio
async def test_hidden_dict_cannot_toggle_studying(fresh_db):
    uid, _did = await pytest.seed_user()
    h = await get_or_create_hidden_dict(uid)
    res = await set_dictionary_studying(uid, h, False)
    assert res.get("error")


@pytest.mark.asyncio
async def test_set_studying_unknown_dict(fresh_db):
    uid, _did = await pytest.seed_user()
    res = await set_dictionary_studying(uid, 99999, False)
    assert res.get("error") == "Not found"


@pytest.mark.asyncio
async def test_suggest_words_targets_hidden_dict(fresh_db):
    uid, did = await pytest.seed_user()
    # наполняем пул кандидатами уровня A1
    for n in ("ord", "bok", "dag", "natt", "vann"):
        await _add_pool_word(n)
    from db.learning import suggest_words
    res = await suggest_words(uid, count=3, level="A1")
    assert res["added"] >= 1
    assert res["dict"] == HIDDEN_DICT_NAME

    # добавленные слова попали в скрытый словарь, не в личный
    data = await get_user_data(uid)
    personal_words = sum(len(d["words"]) for d in data["dictList"])
    assert personal_words == 0  # ничего не попало в видимые словари

    # но в Учёбе они есть
    assert len(await _learning_pools(uid)) >= res["added"]
