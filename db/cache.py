import json
from .core import _conn, _now, normalize_query


# --- Кэш запросов генерации ---
async def get_cached_query(query: str):
    key = normalize_query(query)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT response FROM query_cache WHERE query = ?", (key,)) as cur:
            row = await cur.fetchone()
            return json.loads(row["response"]) if row else None
    finally:
        await db.close()


async def cache_query(query: str, response):
    key = normalize_query(query)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("INSERT OR IGNORE INTO query_cache (query, response, created_at) VALUES (?, ?, ?)",
                         (key, json.dumps(response, ensure_ascii=False), _now()))
        await db.commit()
    finally:
        await db.close()


async def clear_query_cache():
    db = await _conn()
    try:
        await db.execute("DELETE FROM query_cache")
        await db.commit()
    finally:
        await db.close()


# --- Дневной учёт обращений к LLM ---
async def get_usage(day: str) -> int:
    db = await _conn()
    try:
        async with db.execute("SELECT n FROM usage WHERE day = ?", (day,)) as cur:
            r = await cur.fetchone()
            return r["n"] if r else 0
    finally:
        await db.close()


async def incr_usage(day: str, by: int = 1) -> int:
    db = await _conn()
    try:
        await db.execute(
            "INSERT INTO usage (day, n) VALUES (?, ?) ON CONFLICT(day) DO UPDATE SET n = n + ?",
            (day, by, by),
        )
        await db.commit()
        async with db.execute("SELECT n FROM usage WHERE day = ?", (day,)) as cur:
            return (await cur.fetchone())["n"]
    finally:
        await db.close()
