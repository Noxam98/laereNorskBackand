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


async def set_user_password(user_id: int, hashed_password: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
        await db.commit()
    finally:
        await _release(db)


async def get_user_by_google_sub(google_sub: str):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await _release(db)


async def set_online_prefs(user_id: int, prefs_json: str):
    """Запомнить последние настройки онлайн-комнаты (чтобы не настраивать каждый раз)."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET online_prefs = ? WHERE id = ?", (prefs_json, user_id))
        await db.commit()
    finally:
        await _release(db)


async def save_match(game: str, data_json: str):
    """Сохранить результат онлайн-матча в match_log."""
    db = await _conn()
    try:
        await db.execute("INSERT INTO match_log (game, created_at, data) VALUES (?, ?, ?)",
                         (game, _now(), data_json))
        await db.commit()
    finally:
        await _release(db)


async def set_user_name(user_id: int, name: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET display_name = ? WHERE id = ?", (name, user_id))
        await db.commit()
    finally:
        await _release(db)


async def create_google_user(username: str, email: str, google_sub: str, display_name: str = None):
    """Новый аккаунт через Google: пароля нет ('' — bcrypt его не примет), есть email/google_sub."""
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO users (username, password, email, google_sub, display_name) VALUES (?, '', ?, ?, ?)",
            (username, email, google_sub, display_name or None),
        )
        user_id = cur.lastrowid
        await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, "default", _now()))
        await db.commit()
        return {"user_id": user_id}
    except aiosqlite.IntegrityError:
        return {"error": "User already exists"}
    finally:
        await _release(db)


async def set_user_google(user_id: int, google_sub: str, email: str):
    """Привязать Google к существующему аккаунту. IntegrityError, если этот sub уже занят."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET google_sub = ?, email = ? WHERE id = ?", (google_sub, email, user_id))
        await db.commit()
    finally:
        await _release(db)


async def clear_user_google(user_id: int):
    """Отвязать Google (email оставляем как контакт)."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET google_sub = NULL WHERE id = ?", (user_id,))
        await db.commit()
    finally:
        await _release(db)
