"""Починки разблокировки композитов и целостности данных вокруг word_pool_compounds.

#1 омоним-композит открывается по pool_id (а не по написанию — одна запись не отсекает вторую);
#2 часть композита с заглавной матчится после нормализации в lower (банк + рантайм-чтение индекса);
#3 TTL-кеш индекса инвалидируется при записи (не залипает на устаревшем снимке);
#4 merge_pool_words чистит word_pool_compounds/ремапит form_srs (нет фантомного unlock/потери форм).
"""
import json
import sqlite3

from db.core import _conn, _release
from db import compound_index, ordbank
from db.compound_index import compounds_unlocked_by
from db.learning_suggest import suggest_compounds, unlocked_compounds_count
from db.pool_dedup import merge_pool_words
from tests.conftest import seed_user


async def _seed(dbc, did, uid, no, *, pos="noun", mastered=False, in_dict=True, freq=3.0):
    """Слово в пул (+ опц. в словарь юзера / mastered). Вернуть pool_id.
    pos прокинут в колонку и в data — для омонимов (UNIQUE(norwegian, pos))."""
    cur = await dbc.execute(
        "INSERT INTO word_pool (norwegian, data, pos, level, freq, created_at) "
        "VALUES (?,?,?,?,?,datetime('now'))",
        (no, json.dumps({"translate": {"ru": [no]}, "part_of_speech": pos}), pos, "A1", freq))
    pid = cur.lastrowid
    if in_dict:
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
    if mastered:
        await dbc.execute(
            "INSERT INTO user_words (user_id, pool_id, mastered, correct, created_at) "
            "VALUES (?,?,1,4,datetime('now'))", (uid, pid))
    return pid


# ── #1: омоним-композит разблокируется по pool_id ────────────────────────────
async def test_homograph_compound_unlocks_by_pool_id(fresh_db):
    """Композит-ОМОНИМ (одно написание, две записи пула, один в словаре юзера — другой нет):
    владение первой записью НЕ должно отсекать вторую (сверка have по pool_id, а не по строке)."""
    uid, did = await seed_user("hg")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "kjøle", mastered=True)
        await _seed(dbc, did, uid, "skap", mastered=True)
        c1 = await _seed(dbc, did, uid, "kjøleskap", pos="noun", in_dict=True, freq=4.0)   # уже у юзера
        c2 = await _seed(dbc, did, uid, "kjøleskap", pos="verb", in_dict=False, freq=4.0)  # ещё нет
        await dbc.commit()
    finally:
        await _release(dbc)
    compound_index._CACHE["index"] = None
    await compound_index.set_pool_compounds([
        (c1, "kjøleskap", "kjøle", "skap"),
        (c2, "kjøleskap", "kjøle", "skap"),
    ])
    # владение c1 (то же написание) НЕ отсекает c2 — вторая запись открыта
    assert await unlocked_compounds_count(uid) == 1
    assert (await suggest_compounds(uid))["added"] == 1
    assert await unlocked_compounds_count(uid) == 0     # c2 добавлен → больше не «ожидает»


# ── #2: часть с заглавной матчится после .lower() ────────────────────────────
def test_ordbank_compound_lowercases_parts(tmp_path):
    """ordbank.compound() нормализует forledd/etterledd/parts в lower (старые дампы клали
    части с заглавной; pool.norwegian всегда lower → иначе не совпадёт)."""
    p = tmp_path / "ordbank.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE compounds (norwegian TEXT PRIMARY KEY, forledd TEXT NOT NULL, "
                "fuge TEXT, etterledd TEXT NOT NULL, marked TEXT) WITHOUT ROWID")
    con.execute("INSERT INTO compounds VALUES (?,?,?,?,?)",
                ("kjøleskap", "Kjøle", "", "Skap", "kjøle-skap"))
    con.commit(); con.close()
    old_path, old_conn = ordbank.PATH, ordbank._conn
    ordbank.PATH, ordbank._conn = str(p), None
    try:
        c = ordbank.compound("kjøleskap")
        assert c["forledd"] == "kjøle" and c["etterledd"] == "skap"
        assert c["parts"] == ["kjøle", "skap"]
        assert c["marked"] == "kjøle-skap"          # marked — для показа, остаётся как есть
    finally:
        ordbank.PATH, ordbank._conn = old_path, old_conn


async def test_uppercase_parts_in_index_still_unlock(fresh_db):
    """Существующие строки word_pool_compounds с заглавными частями (старый дамп) всё равно
    открывают композит — load_index/compounds_unlocked_by приводят части к lower на чтении."""
    uid, did = await seed_user("uc")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "kjøle", mastered=True)
        await _seed(dbc, did, uid, "skap", mastered=True)
        cpid = await _seed(dbc, did, uid, "kjøleskap", in_dict=False, freq=4.0)
        # прямая вставка С ЗАГЛАВНЫМИ частями (в обход set_pool_compounds, который бы их нормализовал)
        await dbc.execute("INSERT INTO word_pool_compounds VALUES (?,?,?,?)",
                          (cpid, "kjøleskap", "Kjøle", "Skap"))
        await dbc.commit()
    finally:
        await _release(dbc)
    compound_index.invalidate()
    assert await unlocked_compounds_count(uid) == 1
    dbc = await _conn()
    try:
        assert await compounds_unlocked_by(dbc, uid, "skap") == 1   # выучил skap → открыл kjøleskap
    finally:
        await _release(dbc)


# ── #3: инвалидация TTL-кеша при записи ──────────────────────────────────────
async def test_set_pool_compounds_invalidates_cache(fresh_db):
    """Кеш индекса не залипает: после первого load_index (пустой индекс) запись нового композита
    инвалидирует кеш и он сразу виден, не дожидаясь TTL."""
    uid, did = await seed_user("cache")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "kjøle", mastered=True)
        await _seed(dbc, did, uid, "skap", mastered=True)
        cpid = await _seed(dbc, did, uid, "kjøleskap", in_dict=False, freq=4.0)
        await dbc.commit()
    finally:
        await _release(dbc)
    compound_index._CACHE["index"] = None
    # прогреваем кеш при ПУСТОМ индексе (композит ещё не зарегистрирован)
    assert await unlocked_compounds_count(uid) == 0
    assert compound_index._CACHE["index"] == []
    # регистрируем композит → set_pool_compounds инвалидирует → сразу виден (не устаревший 0)
    await compound_index.set_pool_compounds([(cpid, "kjøleskap", "kjøle", "skap")])
    assert compound_index._CACHE["index"] is None
    assert await unlocked_compounds_count(uid) == 1


def test_invalidate_bumps_generation():
    """invalidate() сбрасывает снимок и двигает поколение (страж от перезаписи свежей инвалидации)."""
    compound_index._CACHE.update(index=[{"x": 1}], at=1e9)
    g0 = compound_index._GEN
    compound_index.invalidate()
    assert compound_index._CACHE["index"] is None
    assert compound_index._GEN == g0 + 1


# ── #4: merge_pool_words чистит word_pool_compounds и ремапит form_srs ────────
async def test_merge_cleans_compounds_and_remaps_form_srs(fresh_db):
    """Слияние дублей: строка loser в word_pool_compounds не должна повиснуть (нет FK) —
    иначе фантомный unlock; form_srs loser'а ремапится на winner (а не теряется по CASCADE)."""
    uid, did = await seed_user("mrg")
    dbc = await _conn()
    try:
        await _seed(dbc, did, uid, "kjøle", mastered=True)
        await _seed(dbc, did, uid, "skap", mastered=True)
        winner = await _seed(dbc, did, uid, "kjøleskap", pos="noun", in_dict=False, freq=4.0)
        loser = await _seed(dbc, did, uid, "kjøleskap", pos="verb", in_dict=False, freq=4.0)
        # у loser есть строка индекса и прогресс формы
        await dbc.execute("INSERT INTO word_pool_compounds VALUES (?,?,?,?)",
                          (loser, "kjøleskap", "kjøle", "skap"))
        await dbc.execute("INSERT INTO form_srs (user_id, pool_id, cell) VALUES (?,?,?)",
                          (uid, loser, "def_sg"))
        await dbc.commit()
    finally:
        await _release(dbc)

    assert await merge_pool_words(winner, loser) is True

    dbc = await _conn()
    try:
        # висячей строки loser нет; разбор переехал на winner (тот же композит)
        async with dbc.execute("SELECT pool_id FROM word_pool_compounds WHERE norwegian='kjøleskap'") as cur:
            pids = {r["pool_id"] for r in await cur.fetchall()}
        assert loser not in pids and winner in pids
        # form_srs ремапнут на winner, loser не осталось
        async with dbc.execute("SELECT pool_id FROM form_srs WHERE user_id=?", (uid,)) as cur:
            fpids = {r["pool_id"] for r in await cur.fetchall()}
        assert winner in fpids and loser not in fpids
    finally:
        await _release(dbc)
