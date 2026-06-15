import aiosqlite
import os
from datetime import datetime

# Путь к БД через env (на Fly указывает на смонтированный volume, напр. /data/users.db).
DATABASE_URL = os.getenv("DATABASE_PATH", "users.db")


def _now():
    return datetime.utcnow().isoformat()


async def _conn():
    db = await aiosqlite.connect(DATABASE_URL)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    return db


def normalize_word(norwegian: str) -> str:
    return (norwegian or "").strip().lower()


def normalize_query(query: str) -> str:
    return " ".join((query or "").strip().lower().split())


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
        # уровень CEFR (A1..C2); NULL = ещё не классифицировано
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN level TEXT")
        except Exception:
            pass

        # Теги-темы общего пула (много на слово).
        await db.execute("""
        CREATE TABLE IF NOT EXISTS word_topics (
            pool_id INTEGER NOT NULL,
            topic   TEXT    NOT NULL,
            PRIMARY KEY (pool_id, topic),
            FOREIGN KEY (pool_id) REFERENCES word_pool(id) ON DELETE CASCADE
        )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_word_topics_topic ON word_topics(topic)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_word_pool_level ON word_pool(level)")

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
