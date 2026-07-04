"""Разбор составных слов (sammensetning) из банка: аксессор ordbank.compound + отдача в meta.

leddanalyse Norsk ordbank даёт готовый разбор (forledd/fuge/etterledd). Аксессор
читает таблицу compounds; старый банк без неё деградирует в None. get_pool_meta
пробрасывает разбор в карточку слова.
"""
import sqlite3

import pytest

from db import ordbank


@pytest.fixture
def bank_with_compounds(tmp_path):
    """Временный ordbank.db с таблицей compounds; PATH подменён, кеш соединения сброшен."""
    p = tmp_path / "ordbank.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE compounds (norwegian TEXT PRIMARY KEY, forledd TEXT NOT NULL, "
                "fuge TEXT, etterledd TEXT NOT NULL, marked TEXT) WITHOUT ROWID")
    con.executemany("INSERT INTO compounds VALUES (?,?,?,?,?)", [
        ("barnehage", "barn", "e", "hage", "barne-hage"),
        ("arbeidsplass", "arbeid", "s", "plass", "arbeids-plass"),
        ("flyplass", "fly", "", "plass", "fly-plass"),
        ("barnehagelærer", "barnehage", "", "lærer", "barnehage-lærer"),
    ])
    con.commit()
    con.close()
    old_path, old_conn = ordbank.PATH, ordbank._conn
    ordbank.PATH, ordbank._conn = str(p), None
    yield
    ordbank.PATH, ordbank._conn = old_path, old_conn


def test_compound_lookup_returns_parts(bank_with_compounds):
    c = ordbank.compound("barnehage")
    assert c == {"forledd": "barn", "fuge": "e", "etterledd": "hage",
                 "marked": "barne-hage", "parts": ["barn", "hage"]}
    assert ordbank.compound("flyplass")["fuge"] == ""       # без соединителя
    assert ordbank.compound("arbeidsplass")["parts"] == ["arbeid", "plass"]


def test_compound_recursive_head_is_itself_compound(bank_with_compounds):
    # части — самостоятельные леммы: голова barnehagelærer сама составная
    top = ordbank.compound("barnehagelærer")
    assert top["parts"] == ["barnehage", "lærer"]
    assert ordbank.compound(top["parts"][0])["parts"] == ["barn", "hage"]


def test_compound_case_and_whitespace_insensitive(bank_with_compounds):
    assert ordbank.compound(" BarneHage ")["forledd"] == "barn"


def test_non_compound_returns_none(bank_with_compounds):
    assert ordbank.compound("universitet") is None          # нет в таблице → не составное
    assert ordbank.compound("") is None


def test_missing_bank_degrades_to_none(tmp_path):
    old_path, old_conn = ordbank.PATH, ordbank._conn
    ordbank.PATH, ordbank._conn = str(tmp_path / "nope.db"), None
    try:
        assert ordbank.compound("barnehage") is None        # файла нет → None, не падение
    finally:
        ordbank.PATH, ordbank._conn = old_path, old_conn


async def test_get_pool_meta_carries_compound_key(fresh_db):
    """get_pool_meta всегда несёт ключ compound (None без банка) — контракт карточки."""
    import json
    from db.core import _conn, _release
    from db.pool import get_pool_meta
    dbc = await _conn()
    try:
        await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            ("hus", json.dumps({"translate": {"ru": ["дом"]}, "part_of_speech": "noun"}), "noun", "A1"))
        await dbc.commit()
    finally:
        await _release(dbc)
    meta = await get_pool_meta("hus")
    assert "compound" in meta and meta["compound"] is None    # банка в тестах нет → None
