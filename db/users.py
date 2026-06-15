import aiosqlite
from .core import _conn, _now


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
