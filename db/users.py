import aiosqlite
from .core import _conn, _release, _now


async def get_user(username: str):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await _release(db)


async def set_user_theme(user_id: int, theme: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET theme = ? WHERE id = ?", (theme, user_id))
        await db.commit()
    finally:
        await _release(db)


async def set_user_game_prefs(user_id: int, prefs_json: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_prefs = ? WHERE id = ?", (prefs_json, user_id))
        await db.commit()
    finally:
        await _release(db)


async def set_user_current_dict(user_id: int, name: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET current_dict = ? WHERE id = ?", (name, user_id))
        await db.commit()
    finally:
        await _release(db)


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
        await _release(db)
