import json
from .core import _conn, _release, _now, normalize_query


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
        await _release(db)


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
        await _release(db)


async def set_cached_query(query: str, response):
    """Перезаписать кэш (для принудительной перегенерации)."""
    key = normalize_query(query)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("INSERT OR REPLACE INTO query_cache (query, response, created_at) VALUES (?, ?, ?)",
                         (key, json.dumps(response, ensure_ascii=False), _now()))
        await db.commit()
    finally:
        await _release(db)


async def clear_query_cache():
    db = await _conn()
    try:
        await db.execute("DELETE FROM query_cache")
        await db.commit()
    finally:
        await _release(db)


# --- Дневной учёт обращений к LLM ---
async def get_usage_like(prefix: str):
    """{ключ: n} для всех записей usage, начинающихся с prefix (напр. сегодняшняя дата)."""
    db = await _conn()
    try:
        async with db.execute("SELECT day, n FROM usage WHERE day LIKE ? ORDER BY day", (prefix + "%",)) as cur:
            return {r["day"]: r["n"] for r in await cur.fetchall()}
    finally:
        await _release(db)


async def get_usage(day: str) -> int:
    db = await _conn()
    try:
        async with db.execute("SELECT n FROM usage WHERE day = ?", (day,)) as cur:
            r = await cur.fetchone()
            return r["n"] if r else 0
    finally:
        await _release(db)


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
        await _release(db)
