import aiosqlite

DATABASE_URL = "users.db"

async def init_db():
    async with aiosqlite.connect(DATABASE_URL) as db:
        await db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
        """)
        await db.commit()

async def get_user(username: str):
    async with aiosqlite.connect(DATABASE_URL) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cursor:
            user = await cursor.fetchone()
            return dict(user) if user else None

async def create_user(username: str, hashed_password: str):
    async with aiosqlite.connect(DATABASE_URL) as db:
        try:
            await db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            await db.commit()
            return {"message": "User created successfully"}
        except aiosqlite.IntegrityError:
            return {"error": "Username already exists"}
