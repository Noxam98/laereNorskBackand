"""Связь словаря и Учёбы (§7): флаги studying/hidden на словарях.

Что проверяем (осмысленно, через публичные функции, а не подгонка под реализацию):
  (1) миграция: у dictionaries есть колонки hidden/studying; дефолты studying=1, hidden=0;
  (2) _fetch_user_words: слова из словаря со studying=0 НЕ попадают в Учёбу, со studying=1 — попадают;
  (3) suggest_words/seed_starter кладут слова в СКРЫТЫЙ авто-словарь (hidden=1, studying=1),
      не в личный; повторный вызов переиспользует тот же скрытый словарь (не плодит);
  (4) get_user_data НЕ содержит скрытый словарь (hidden=1), но личные — содержит;
  (5) set_dictionary_studying переключает флаг; после выключения слова словаря выпадают из Учёбы.
"""
import json
import pytest

from db.core import _conn, _release, _now
from db.dictionaries import (
    add_word_to_dict, get_user_data, get_or_create_hidden_dict,
    set_dictionary_studying, HIDDEN_DICT_NAME,
)
from db.learning import _fetch_user_words, suggest_words, seed_starter
from tests.conftest import seed_user, seed_word


# ---------------- хелперы ----------------

async def _add_pool_word(no, ru="перевод", level="A1"):
    """Добавить слово только в общий пул (для кандидатов suggest_words). Вернуть pool_id."""
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": "noun"})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
            (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _learning_pools(user_id):
    """Множество pool_id, которые Учёба видит для пользователя."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        return {r["pool_id"] for r in rows}
    finally:
        await _release(db)


async def _dict_flags(dict_id):
    """(hidden, studying) словаря как они лежат в БД."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT hidden, studying FROM dictionaries WHERE id = ?", (dict_id,)) as cur:
            row = await cur.fetchone()
        return (row["hidden"], row["studying"]) if row else (None, None)
    finally:
        await _release(db)


async def _hidden_dict_count(user_id):
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(*) AS n FROM dictionaries WHERE user_id = ? AND COALESCE(hidden,0) = 1",
            (user_id,)) as cur:
            return (await cur.fetchone())["n"]
    finally:
        await _release(db)


# ---------------- (1) миграция и дефолты ----------------

@pytest.mark.asyncio
async def test_migration_columns_and_defaults(fresh_db):
    # колонки существуют в схеме
    db = await _conn()
    try:
        async with db.execute("PRAGMA table_info(dictionaries)") as cur:
            cols = {r["name"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert "hidden" in cols
    assert "studying" in cols

    # seed_user создаёт обычный словарь без явного указания флагов → дефолты
    _uid, did = await seed_user()
    hidden, studying = await _dict_flags(did)
    assert studying == 1   # по умолчанию словарь участвует в Учёбе
    assert hidden == 0     # по умолчанию словарь виден в «Мой словарь»


# ---------------- (2) _fetch_user_words уважает studying ----------------

@pytest.mark.asyncio
async def test_fetch_user_words_respects_studying_flag(fresh_db):
    uid, did_on = await seed_user()
    pid_on, _ = await seed_word(did_on, "hus", "дом")

    # второй личный словарь, выключенный из обучения
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO dictionaries (user_id,name,created_at,studying) VALUES (?,?,?,0)",
            (uid, "off", _now()))
        did_off = cur.lastrowid
        await db.commit()
    finally:
        await _release(db)
    pid_off, _ = await seed_word(did_off, "katt", "кот")

    pools = await _learning_pools(uid)
    assert pid_on in pools        # studying=1 → в Учёбе
    assert pid_off not in pools   # studying=0 → не в Учёбе


# ---------------- (3) suggest_words/seed_starter → скрытый авто-словарь, переиспользование ----------------

@pytest.mark.asyncio
async def test_suggest_words_targets_hidden_dict_and_reuses_it(fresh_db):
    uid, _did = await seed_user()
    # кандидаты уровня A1 в пуле
    for n in ("ord", "bok", "dag", "natt", "vann", "sol", "hav", "fjell"):
        await _add_pool_word(n)

    res1 = await suggest_words(uid, count=3, level="A1")
    assert res1["added"] >= 1
    assert res1["dict"] == HIDDEN_DICT_NAME

    # ровно один скрытый словарь, и он hidden=1, studying=1
    assert await _hidden_dict_count(uid) == 1
    hid = await get_or_create_hidden_dict(uid)
    hidden, studying = await _dict_flags(hid)
    assert hidden == 1 and studying == 1

    # ничего не попало в видимые (личные) словари
    data = await get_user_data(uid)
    assert sum(len(d["words"]) for d in data["dictList"]) == 0

    # но слова видны Учёбе
    assert len(await _learning_pools(uid)) >= res1["added"]

    # повторный вызов НЕ плодит скрытые словари — переиспользует тот же
    res2 = await suggest_words(uid, count=2, level="A1")
    assert res2["added"] >= 1
    assert await _hidden_dict_count(uid) == 1
    assert await get_or_create_hidden_dict(uid) == hid


@pytest.mark.asyncio
async def test_seed_starter_targets_hidden_dict(fresh_db):
    uid, _did = await seed_user()
    for n in ("en", "to", "tre", "fire", "fem", "seks", "sju", "atte", "ni", "ti"):
        await _add_pool_word(n)

    res = await seed_starter(uid, "A1", target=5)
    assert res["seeded"] >= 1

    # всё ушло в скрытый словарь, личные пусты, один скрытый
    assert await _hidden_dict_count(uid) == 1
    data = await get_user_data(uid)
    assert sum(len(d["words"]) for d in data["dictList"]) == 0
    assert len(await _learning_pools(uid)) >= res["seeded"]


# ---------------- (4) get_user_data: скрытый скрыт, личные видны ----------------

@pytest.mark.asyncio
async def test_get_user_data_excludes_hidden_includes_personal(fresh_db):
    uid, did = await seed_user()
    pid_personal, _ = await seed_word(did, "bil", "машина")

    h = await get_or_create_hidden_dict(uid)
    pid_hidden = await _add_pool_word("sykkel", "велосипед")
    await add_word_to_dict(uid, h, pid_hidden)

    data = await get_user_data(uid)
    names = [d["dictName"] for d in data["dictList"]]
    assert HIDDEN_DICT_NAME not in names          # скрытый не показываем
    assert "default" in names                     # личный словарь виден
    # личное слово присутствует, скрытое — нет (в «Мой словарь»)
    personal_pool_present = any(d["dictName"] == "default" and d["words"] for d in data["dictList"])
    assert personal_pool_present
    assert all("studying" in d for d in data["dictList"])  # флаг отдаётся фронту


# ---------------- (5) set_dictionary_studying переключает; выкл → выпадают из Учёбы ----------------

@pytest.mark.asyncio
async def test_set_studying_toggles_and_drops_from_learning(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "skole", "школа")
    assert pid in await _learning_pools(uid)

    res_off = await set_dictionary_studying(uid, did, False)
    assert res_off["studying"] is False
    assert (await _dict_flags(did))[1] == 0
    assert pid not in await _learning_pools(uid)   # выпало из Учёбы

    res_on = await set_dictionary_studying(uid, did, True)
    assert res_on["studying"] is True
    assert (await _dict_flags(did))[1] == 1
    assert pid in await _learning_pools(uid)        # вернулось
