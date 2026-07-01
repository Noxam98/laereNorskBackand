"""D4: init_db как МИГРАЦИЯ поверх существующей БД (не с нуля) — раньше не тестировалось (fresh_db
всегда строит схему с нуля), а blast radius — прод-volume с реальными данными. Проверяем: ALTER
добавляет новые колонки к старой таблице, данные целы, повторный init_db не падает (идемпотентность)."""
import os
import tempfile

import aiosqlite

from db import core
from db.core import init_db, _conn, _release


async def test_init_db_migrates_existing_db():
    orig_url = core.DATABASE_URL
    path = tempfile.mktemp(suffix=".db")
    # «Старая» БД: word_pool без новых колонок (freq/pos/forms/embedding/approved/…), с данными.
    db = await aiosqlite.connect(path)
    await db.execute("CREATE TABLE word_pool (id INTEGER PRIMARY KEY, norwegian TEXT, data TEXT, created_at TEXT)")
    await db.execute("INSERT INTO word_pool (norwegian, data, created_at) VALUES ('gammelord', '{}', '2020-01-01')")
    await db.commit()
    await db.close()

    core.DATABASE_URL = path
    try:
        await init_db()                        # миграция поверх существующей БД

        dbc = await _conn()
        try:
            async with dbc.execute("PRAGMA table_info(word_pool)") as cur:
                cols = {r["name"] for r in await cur.fetchall()}
            assert {"freq", "pos", "forms", "embedding", "approved"} <= cols   # новые колонки добавлены
            async with dbc.execute("SELECT norwegian FROM word_pool WHERE norwegian='gammelord'") as cur:
                row = await cur.fetchone()
            assert row and row["norwegian"] == "gammelord"                     # старые данные целы
        finally:
            await _release(dbc)

        await init_db()                        # идемпотентность: повторный прогон не падает
    finally:
        core.DATABASE_URL = orig_url
        try:
            os.remove(path)
        except OSError:
            pass
