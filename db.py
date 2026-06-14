import aiosqlite
import json
import os
from datetime import datetime

# Путь к БД через env (на Fly указывает на смонтированный volume, напр. /data/users.db).
DATABASE_URL = os.getenv("DATABASE_PATH", "users.db")


async def init_db():
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """)

        # Общий пул сгенерированных слов: каждое норвежское слово хранится один раз
        # и переиспользуется всеми пользователями.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS word_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            norwegian TEXT UNIQUE NOT NULL,
            data TEXT NOT NULL,
            description TEXT,
            embedding TEXT,
            created_at TEXT NOT NULL
        )
        """)
        # миграция для старых БД
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN embedding TEXT")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN tts BLOB")
        except Exception:
            pass

        # Кэш запросов генерации.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS query_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """)

        # Дневной учёт обращений к LLM (для авто-заполнения пула в рамках бюджета).
        await db.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            day TEXT PRIMARY KEY,
            n INTEGER NOT NULL DEFAULT 0
        )
        """)

        # Словари пользователя.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS dictionaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, name),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)

        # Слова в словаре пользователя: ссылка на общий пул + персональные правки и статистика.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS dict_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dict_id INTEGER NOT NULL,
            pool_id INTEGER NOT NULL,
            override TEXT,
            correct INTEGER NOT NULL DEFAULT 0,
            incorrect INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(dict_id, pool_id),
            FOREIGN KEY(dict_id) REFERENCES dictionaries(id) ON DELETE CASCADE,
            FOREIGN KEY(pool_id) REFERENCES word_pool(id) ON DELETE CASCADE
        )
        """)
        await db.commit()


def _now():
    return datetime.utcnow().isoformat()


async def _conn():
    db = await aiosqlite.connect(DATABASE_URL)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    return db


# --- Пользователи ---
async def get_user(username: str):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await db.close()


async def create_user(username: str, hashed_password: str):
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        user_id = cur.lastrowid
        # стартовый словарь
        await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, "default", _now()))
        await db.commit()
        return {"message": "User created successfully", "user_id": user_id}
    except aiosqlite.IntegrityError:
        return {"error": "Username already exists"}
    finally:
        await db.close()


# --- Общий пул слов ---
def normalize_word(norwegian: str) -> str:
    return (norwegian or "").strip().lower()


async def get_or_create_pool(norwegian: str, data: dict):
    """Вернуть id записи пула для норвежского слова, создав её при необходимости."""
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            row = await cur.fetchone()
            if row:
                return row["id"]
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, ensure_ascii=False), _now()),
        )
        await db.commit()
        return cur.lastrowid
    finally:
        await db.close()


async def get_pool_id(norwegian: str):
    """id записи пула по норвежскому слову (без создания)."""
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["id"] if r else None
    finally:
        await db.close()


async def get_pool_by_id(pool_id: int):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "data": json.loads(row["data"]),
                "description": json.loads(row["description"]) if row["description"] else None,
                "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
            }
    finally:
        await db.close()


async def get_pool_tts(norwegian: str):
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT tts FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["tts"] if r and r["tts"] else None
    finally:
        await db.close()


async def set_pool_tts(norwegian: str, data: bytes):
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET tts = ? WHERE norwegian = ?", (data, key))
        await db.commit()
    finally:
        await db.close()


async def set_pool_embedding(pool_id: int, vector: list):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET embedding = ? WHERE id = ?", (json.dumps(vector), pool_id))
        await db.commit()
    finally:
        await db.close()


async def get_pool_candidates():
    """Все слова пула (id, norwegian, data, embedding) — для подбора дистракторов."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian, data, embedding FROM word_pool") as cur:
            rows = await cur.fetchall()
            out = []
            for r in rows:
                out.append({
                    "id": r["id"],
                    "norwegian": r["norwegian"],
                    "data": json.loads(r["data"]) if r["data"] else {},
                    "embedding": json.loads(r["embedding"]) if r["embedding"] else None,
                })
            return out
    finally:
        await db.close()


async def set_pool_description(pool_id: int, description: dict):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET description = ? WHERE id = ?", (json.dumps(description, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await db.close()


# --- Кэш запросов генерации ---
def normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


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


# --- Словари и слова пользователя ---
async def create_dictionary(user_id: int, name: str):
    name = (name or "").strip()
    if not name:
        return {"error": "Empty name"}
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, name, _now()))
        await db.commit()
        return {"id": cur.lastrowid, "name": name}
    except aiosqlite.IntegrityError:
        return {"error": "Dictionary already exists"}
    finally:
        await db.close()


async def delete_dictionary(user_id: int, dict_id: int):
    db = await _conn()
    try:
        # не удалять последний словарь
        async with db.execute("SELECT COUNT(*) c FROM dictionaries WHERE user_id = ?", (user_id,)) as cur:
            if (await cur.fetchone())["c"] <= 1:
                return {"error": "Cannot delete the last dictionary"}
        await db.execute("DELETE FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await db.close()


async def _owns_dict(db, user_id, dict_id):
    async with db.execute("SELECT id FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id)) as cur:
        return (await cur.fetchone()) is not None


async def add_word_to_dict(user_id: int, dict_id: int, pool_id: int):
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, dict_id):
            return {"error": "Not found"}
        try:
            cur = await db.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?, ?, ?)", (dict_id, pool_id, _now()))
            await db.commit()
            return {"id": cur.lastrowid}
        except aiosqlite.IntegrityError:
            # уже есть в этом словаре
            async with db.execute("SELECT id FROM dict_words WHERE dict_id = ? AND pool_id = ?", (dict_id, pool_id)) as cur:
                row = await cur.fetchone()
                return {"id": row["id"] if row else None, "duplicate": True}
    finally:
        await db.close()


async def delete_dict_word(user_id: int, dw_id: int):
    db = await _conn()
    try:
        await db.execute("""
            DELETE FROM dict_words WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await db.close()


async def set_word_override(user_id: int, dw_id: int, override: dict):
    db = await _conn()
    try:
        await db.execute("""
            UPDATE dict_words SET override = ?
            WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (json.dumps(override, ensure_ascii=False), dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await db.close()


async def record_result(user_id: int, dw_id: int, correct: bool):
    db = await _conn()
    try:
        col = "correct" if correct else "incorrect"
        await db.execute(f"""
            UPDATE dict_words SET {col} = {col} + 1
            WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await db.close()


async def delete_pool_word(norwegian: str):
    """Полностью удалить слово из общего пула (каскадом у всех) + почистить кэш запросов."""
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("DELETE FROM word_pool WHERE norwegian = ?", (key,))
        await db.execute(
            "DELETE FROM query_cache WHERE query LIKE ? OR response LIKE ? OR response LIKE ?",
            (f"%{key}%", f"%{key}%", f"%{norwegian}%"),
        )
        await db.commit()
    finally:
        await db.close()


async def get_dict_word(user_id: int, dw_id: int):
    """Вернуть (dw_id, pool_id, dict_id, description, data, embedding) для слова пользователя."""
    db = await _conn()
    try:
        async with db.execute("""
            SELECT dw.id, dw.pool_id, dw.dict_id, dw.override, wp.norwegian, wp.description, wp.data, wp.embedding
            FROM dict_words dw
            JOIN dictionaries d ON d.id = dw.dict_id
            JOIN word_pool wp ON wp.id = dw.pool_id
            WHERE dw.id = ? AND d.user_id = ?
        """, (dw_id, user_id)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await db.close()


def _build_word(row):
    base = json.loads(row["data"]) if row["data"] else {}
    if row["override"]:
        ov = json.loads(row["override"])
        base = {**base, **ov}
        if "translate" in ov:
            base["translate"] = {**base.get("translate", {}), **ov["translate"]}
    desc = json.loads(row["description"]) if row["description"] else None
    return {
        "id": row["dw_id"],
        "word": base.get("word"),
        "translate": base.get("translate", {}),
        "part_of_speech": base.get("part_of_speech", ""),
        "description": {"description": desc} if desc else None,
        "descriptionState": "loaded" if desc else "empty",
        "gameData": {"correctFirstTry": row["correct"], "incorrectFirstTry": row["incorrect"], "isChoosedToGame": False},
        "techData": {"isSelected": False},
    }


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


async def clear_query_cache():
    db = await _conn()
    try:
        await db.execute("DELETE FROM query_cache")
        await db.commit()
    finally:
        await db.close()


async def get_pool_list(limit: int = 60, offset: int = 0, q: str = None):
    """Список слов из общего пула (с поиском и пагинацией)."""
    key = normalize_word(q) if q else None
    where = "WHERE norwegian LIKE ?" if key else ""
    params = (f"%{key}%",) if key else ()
    db = await _conn()
    try:
        async with db.execute(f"SELECT COUNT(*) c FROM word_pool {where}", params) as cur:
            total = (await cur.fetchone())["c"]
        async with db.execute(
            f"SELECT norwegian, data FROM word_pool {where} ORDER BY norwegian LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ) as cur:
            rows = await cur.fetchall()
            words = []
            for r in rows:
                d = json.loads(r["data"]) if r["data"] else {}
                words.append({"word": r["norwegian"], "translate": d.get("translate", {}), "part_of_speech": d.get("part_of_speech", "")})
        return {"total": total, "words": words}
    finally:
        await db.close()


async def search_pool(prefix: str, limit: int = 10):
    """Поиск по общему пулу слов (для будущего автокомплита)."""
    key = normalize_word(prefix)
    if not key:
        return []
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian, data FROM word_pool WHERE norwegian LIKE ? ORDER BY norwegian LIMIT ?",
            (key + "%", limit),
        ) as cur:
            rows = await cur.fetchall()
            out = []
            for r in rows:
                d = json.loads(r["data"]) if r["data"] else {}
                out.append({"word": r["norwegian"], "translate": d.get("translate", {}), "part_of_speech": d.get("part_of_speech", "")})
            return out
    finally:
        await db.close()


async def get_user_data(user_id: int):
    """Полная структура словарей пользователя в формате, который ждёт фронтенд."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, name FROM dictionaries WHERE user_id = ? ORDER BY created_at, id", (user_id,)) as cur:
            dicts = [dict(r) for r in await cur.fetchall()]
        result = []
        for d in dicts:
            async with db.execute("""
                SELECT dw.id AS dw_id, dw.override, dw.correct, dw.incorrect,
                       wp.data, wp.description
                FROM dict_words dw
                JOIN word_pool wp ON wp.id = dw.pool_id
                WHERE dw.dict_id = ?
                ORDER BY dw.id
            """, (d["id"],)) as cur:
                words = [_build_word(r) for r in await cur.fetchall()]
            result.append({"id": d["id"], "dictName": d["name"], "words": words})
        return {"dictList": result, "dictNames": [d["name"] for d in dicts]}
    finally:
        await db.close()
