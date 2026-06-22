import aiosqlite
import os
import asyncio
import json
import logging
from datetime import datetime

logger = logging.getLogger("learnnorsk")

# Путь к БД через env (на Fly указывает на смонтированный volume, напр. /data/users.db).
DATABASE_URL = os.getenv("DATABASE_PATH", "users.db")
EMBED_DIM = int(os.getenv("EMBED_DIM", "3072"))  # размерность векторов в индексе sqlite-vec

# sqlite-vec (ANN-индекс). Если пакет/расширение недоступны — работаем на brute-force.
try:
    import sqlite_vec
    _HAS_VEC_PKG = True
except Exception:
    _HAS_VEC_PKG = False
SQLITE_VEC_OK = False  # станет True, если расширение реально загрузилось и таблица создана


def _now():
    return datetime.utcnow().isoformat()


# --- Пул соединений (переиспользуем, чтобы не плодить потоки aiosqlite на каждый вызов) ---
_POOL = []
_POOL_LOCK = asyncio.Lock()
_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))


async def _make_conn():
    db = await aiosqlite.connect(DATABASE_URL)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    await db.execute("PRAGMA busy_timeout = 5000")   # ждать блокировку до 5с вместо мгновенной ошибки
    await db.execute("PRAGMA synchronous = NORMAL")  # быстрее, безопасно в режиме WAL
    if _HAS_VEC_PKG:
        try:
            await db.enable_load_extension(True)
            await db.load_extension(sqlite_vec.loadable_path())
            await db.enable_load_extension(False)
        except Exception:
            pass  # расширение не загрузилось — этому соединению vec-запросы недоступны
    return db


async def _conn():
    async with _POOL_LOCK:
        if _POOL:
            return _POOL.pop()
    return await _make_conn()


async def _release(db):
    # сбрасываем возможную незавершённую транзакцию перед возвратом в пул
    try:
        await db.rollback()
    except Exception:
        try:
            await db.close()
        except Exception:
            pass
        return
    async with _POOL_LOCK:
        if len(_POOL) < _POOL_SIZE:
            _POOL.append(db)
            return
    await db.close()


async def get_setting(key: str, default=None):
    """Прочитать значение из key-value таблицы настроек (app_settings)."""
    db = await _conn()
    try:
        async with db.execute("SELECT value FROM app_settings WHERE key=?", (key,)) as cur:
            row = await cur.fetchone()
        return row["value"] if row else default
    finally:
        await _release(db)


async def set_setting(key: str, value: str):
    """Записать значение в key-value таблицу настроек (app_settings)."""
    db = await _conn()
    try:
        await db.execute(
            "INSERT INTO app_settings(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
        await db.commit()
    finally:
        await _release(db)


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
        try:
            await db.execute("ALTER TABLE users ADD COLUMN theme TEXT")  # настройка темы оформления
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN game_prefs TEXT")  # последние настройки игры (JSON)
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN current_dict TEXT")  # последний выбранный словарь
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN email TEXT")  # email от Google; NULL у парольных аккаунтов
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")  # Google account id (sub); NULL = не привязан
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN display_name TEXT")  # отображаемое имя (персонализация)
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN online_prefs TEXT")  # последние настройки онлайн-комнаты (JSON)
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN game_mode TEXT")  # последний режим в хабе «Игры» (solo|online)
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE users ADD COLUMN start_level TEXT")  # уровень по входному тесту (старт «Учёбы»)
        except Exception:
            pass
        try:
            # до какого момента притормозить приток новых слов (мягкий тормоз аудита при >THROTTLE забытого), §2.4-B
            await db.execute("ALTER TABLE users ADD COLUMN audit_throttle_until TEXT")
        except Exception:
            pass
        # Лог результатов онлайн-матчей (для статистики/будущего лидерборда).
        await db.execute("""
        CREATE TABLE IF NOT EXISTS match_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game TEXT NOT NULL,
            created_at TEXT NOT NULL,
            data TEXT NOT NULL
        )
        """)
        # один Google-аккаунт = максимум один наш юзер (частичный uniq, NULL не считаются)
        try:
            await db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_google_sub "
                             "ON users(google_sub) WHERE google_sub IS NOT NULL")
        except Exception:
            pass

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
        # флаг: озвучка всех переводов слова сгенерирована (в Tigris). 0/NULL = нет.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN tts_tr_done INTEGER DEFAULT 0")
        except Exception:
            pass
        # флаг: у слова есть переводы на все 5 языков. 0/NULL = нужно догенерировать.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN translate_done INTEGER DEFAULT 0")
        except Exception:
            pass
        # флаг: эмбеддинг построен по СМЫСЛУ (слово+переводы). 0/NULL = нужно пере-эмбеддить.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN emb_sem INTEGER DEFAULT 0")
        except Exception:
            pass
        # грамматические формы по части речи (JSON: {pos, ...формы}). NULL = ещё не сгенерированы.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN forms TEXT")
        except Exception:
            pass
        # флаг: слово проверено фоновым дедупом на дубль-вариант написания. 0/NULL = в очереди.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN dedup_done INTEGER DEFAULT 0")
        except Exception:
            pass
        # частотность слова по корпусу (Zipf 0..8, выше = употребимее). NULL = ещё не проставлена.
        try:
            await db.execute("ALTER TABLE word_pool ADD COLUMN freq REAL")
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
        await db.execute("CREATE INDEX IF NOT EXISTS idx_word_pool_level_freq ON word_pool(level, freq)")

        # Лексикон bokmål (полный частотный словарь) — для автодополнения словами, которых
        # ещё нет в нашем пуле. Сидинг один раз из data/nb_zipf.json. word PRIMARY KEY → префикс.
        await db.execute("CREATE TABLE IF NOT EXISTS nb_lexicon (word TEXT PRIMARY KEY, zipf REAL)")
        try:
            seeded = (await (await db.execute("SELECT COUNT(*) FROM nb_lexicon")).fetchone())[0]
            if not seeded and os.getenv("NB_LEXICON_SEED", "1") == "1":
                import os as _os
                path = _os.path.join(_os.path.dirname(__file__), "..", "data", "nb_zipf.json")
                with open(path, encoding="utf-8") as f:
                    z = json.load(f)
                await db.executemany("INSERT OR IGNORE INTO nb_lexicon(word, zipf) VALUES (?,?)", list(z.items()))
                await db.commit()
                logger.info(f"nb_lexicon seeded: {len(z)} words")
        except Exception as e:
            logger.warning(f"nb_lexicon seed: {e}")

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

        # Простое key-value хранилище настроек рантайма (напр. паузы фоновых задач),
        # чтобы состояние переживало рестарт/передеплой.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS app_settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
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
        # скрытый авто-словарь (для «докинуть»/стартовый набор) — не показываем в «Мой словарь»
        try:
            await db.execute("ALTER TABLE dictionaries ADD COLUMN hidden INTEGER DEFAULT 0")
        except Exception:
            pass
        # флаг «в обучении»: Учёба берёт слова только из словарей со studying=1 (по умолчанию — да)
        try:
            await db.execute("ALTER TABLE dictionaries ADD COLUMN studying INTEGER DEFAULT 1")
        except Exception:
            pass

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
        # «Учёба»: состояние интервальных повторений на пару (пользователь, слово пула).
        # Слова берутся из словарей пользователя; здесь — сила/статус/расписание повторений.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS user_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            pool_id INTEGER NOT NULL,
            strength INTEGER NOT NULL DEFAULT 0,
            reps INTEGER NOT NULL DEFAULT 0,
            lapses INTEGER NOT NULL DEFAULT 0,
            ease REAL NOT NULL DEFAULT 2.5,
            interval_days REAL NOT NULL DEFAULT 0,
            due_at TEXT,
            correct INTEGER NOT NULL DEFAULT 0,
            incorrect INTEGER NOT NULL DEFAULT 0,
            streak INTEGER NOT NULL DEFAULT 0,
            archived INTEGER NOT NULL DEFAULT 0,
            modes TEXT,                       -- JSON: сколько раз верно пройдено в каждом режиме
            last_seen TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, pool_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(pool_id) REFERENCES word_pool(id) ON DELETE CASCADE
        )
        """)
        try:
            await db.execute("ALTER TABLE user_words ADD COLUMN certified INTEGER DEFAULT 0")  # слово сдало зачётный экзамен-ворота
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE user_words ADD COLUMN audit_due TEXT")  # когда слову пора на аудит забывания (§2.4-B)
        except Exception:
            pass
        try:
            # длина последнего интервала аудита в днях — чтобы при успехе срок РОС (prev × рост), §2.4-B
            await db.execute("ALTER TABLE user_words ADD COLUMN audit_interval REAL DEFAULT 0")
        except Exception:
            pass
        try:
            # слово КОГДА-ЛИБО было сертифицировано (прошло ворота). Не сбрасывается при де-сертификации
            # на аудите: служит признаком «забытое на аудите» для замыкания петли забывания (§2.4-B,
            # «вариант A»): доучил такое до mastered → СРАЗУ ре-сертифицируем, минуя ворота.
            await db.execute("ALTER TABLE user_words ADD COLUMN was_certified INTEGER DEFAULT 0")
        except Exception:
            pass
        # Дневная активность «Учёбы» — для стрика, дневной цели, точности и хитмапа.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            user_id INTEGER NOT NULL,
            day TEXT NOT NULL,
            answers INTEGER NOT NULL DEFAULT 0,
            correct INTEGER NOT NULL DEFAULT 0,
            UNIQUE(user_id, day),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        # Кэш cloze-заданий для служебных слов (персонально, из выученных слов юзера).
        # data = JSON [{blank, answer, options:[...]}], 3 предложения. Чистится при reset/удалении.
        await db.execute("""
        CREATE TABLE IF NOT EXISTS cloze_cache (
            user_id INTEGER NOT NULL,
            pool_id INTEGER NOT NULL,
            data TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, pool_id),
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY(pool_id) REFERENCES word_pool(id) ON DELETE CASCADE
        )
        """)
        await db.execute("PRAGMA journal_mode = WAL")  # параллельные чтения + один писатель
        await db.commit()

    # ANN-индекс sqlite-vec (если расширение доступно). Иначе — brute-force.
    await _init_vec()


async def _init_vec():
    """Создать vec-таблицу и наполнить её из существующих эмбеддингов. Безопасно при отсутствии расширения."""
    global SQLITE_VEC_OK
    if not _HAS_VEC_PKG:
        logger.info("sqlite-vec package not installed — similarity falls back to brute-force")
        return
    import numpy as np
    db = await _make_conn()
    try:
        # косинусная метрика (эмбеддинги не нормированы — L2 даёт плохих соседей)
        await db.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_words USING vec0(embedding float[{EMBED_DIM}] distance_metric=cosine)")
        await db.execute("DROP TABLE IF EXISTS vec_pool")  # снести старый L2-индекс, если был
        await db.commit()
        # бэкфилл: добавить векторы слов, которых ещё нет в индексе
        async with db.execute(
            "SELECT id, embedding FROM word_pool WHERE embedding IS NOT NULL "
            "AND id NOT IN (SELECT rowid FROM vec_words)"
        ) as cur:
            rows = await cur.fetchall()
        added = 0
        for r in rows:
            v = _f16_to_f32_bytes(r["embedding"], np)
            if v is not None:
                await db.execute("INSERT OR REPLACE INTO vec_words(rowid, embedding) VALUES (?, ?)", (r["id"], v))
                added += 1
        if added:
            await db.commit()
        SQLITE_VEC_OK = True
        logger.info(f"sqlite-vec ready (cosine, dim={EMBED_DIM}); backfilled {added} vectors")
    except Exception as e:
        SQLITE_VEC_OK = False
        logger.warning(f"sqlite-vec unavailable, brute-force fallback: {e}")
    finally:
        await db.close()


def _f16_to_f32_bytes(raw, np):
    """Хранимый эмбеддинг (float16 bytes / legacy JSON) → float32 bytes для sqlite-vec."""
    try:
        if isinstance(raw, (bytes, bytearray)) and raw[:1] != b"[":
            v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        else:
            import json
            v = np.asarray(json.loads(raw), dtype=np.float32)
        if v.shape[0] != EMBED_DIM:
            return None
        return v.tobytes()
    except Exception:
        return None


async def vec_upsert(pool_id, raw):
    """Добавить/обновить вектор слова в ANN-индексе."""
    if not SQLITE_VEC_OK:
        return
    import numpy as np
    v = _f16_to_f32_bytes(raw, np)
    if v is None:
        return
    db = await _conn()
    try:
        # vec0 (sqlite-vec) не разрешает конфликт по rowid через INSERT OR REPLACE
        # (падает UNIQUE constraint) — поэтому сначала удаляем, потом вставляем.
        await db.execute("DELETE FROM vec_words WHERE rowid = ?", (pool_id,))
        await db.execute("INSERT INTO vec_words(rowid, embedding) VALUES (?, ?)", (pool_id, v))
        await db.commit()
    finally:
        await _release(db)


async def vec_delete(pool_id):
    if not SQLITE_VEC_OK or pool_id is None:
        return
    db = await _conn()
    try:
        await db.execute("DELETE FROM vec_words WHERE rowid = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def vec_nearest_rows(target_raw, k):
    """Top-k ближайших слов пула через ANN-индекс. None — индекс недоступен."""
    if not SQLITE_VEC_OK:
        return None
    import numpy as np
    q = _f16_to_f32_bytes(target_raw, np)
    if q is None:
        return None
    db = await _conn()
    try:
        async with db.execute(
            "SELECT rowid, distance FROM vec_words WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
            (q, k),
        ) as cur:
            knn = await cur.fetchall()
        ids = [r["rowid"] for r in knn]
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        async with db.execute(f"SELECT id, norwegian, data FROM word_pool WHERE id IN ({marks})", ids) as cur:
            by_id = {r["id"]: r for r in await cur.fetchall()}
        return [{"id": i, "norwegian": by_id[i]["norwegian"], "data": by_id[i]["data"]} for i in ids if i in by_id]
    finally:
        await _release(db)
