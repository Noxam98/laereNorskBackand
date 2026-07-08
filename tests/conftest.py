"""Тестовая инфраструктура для SRS-логики «Учёбы».

Важно: переменные окружения выставляются ДО импорта db, т.к. db.core читает
DATABASE_PATH/DB_POOL_SIZE на уровне модуля. Пул=0 → соединения закрываются на release
(нет висящих потоков, процесс pytest завершается чисто). Каждый тест — свежая временная БД.
"""
import os
import tempfile
import asyncio
import json

os.environ["DB_POOL_SIZE"] = "0"
os.environ["NB_LEXICON_SEED"] = "0"   # не сидим 318k-лексикон в тестах
os.environ.setdefault("DATABASE_PATH", tempfile.mktemp(suffix=".db"))

import pytest  # noqa: E402
import pytest_asyncio  # noqa: E402

from db import core  # noqa: E402
from db.core import init_db, _conn, _release, _now  # noqa: E402


@pytest_asyncio.fixture
async def fresh_db():
    """Свежая пустая БД на каждый тест."""
    path = tempfile.mktemp(suffix=".db")
    core.DATABASE_URL = path  # переключаем путь для этого теста
    # пер-юзер локи кэшируются на модуле и привязаны к event loop; у каждого теста свой loop —
    # чистим, чтобы тест не переиспользовал asyncio.Lock из ЗАКРЫТОГО loop прошлого теста (в проде
    # один долгоживущий loop, проблемы нет). Иначе конкурентный apply_(form_)result → «loop is closed».
    try:
        from db.learning import _USER_LOCKS
        _USER_LOCKS.clear()
    except Exception:
        pass
    await init_db()
    yield path
    try:
        os.remove(path)
    except OSError:
        pass


async def seed_user(username="t"):
    """Создать пользователя + дефолтный словарь, вернуть user_id, dict_id.
    start_level='A1' по умолчанию — юзер в сессии считается прошедшим плейсмент (иначе авто-добор
    заблокирован гейтом «не досыпаем до выбора уровня», см. build_session)."""
    dbc = await _conn()
    try:
        cur = await dbc.execute("INSERT INTO users (username,password,start_level) VALUES (?, 'x', 'A1')", (username,))
        uid = cur.lastrowid
        cur = await dbc.execute("INSERT INTO dictionaries (user_id,name,created_at) VALUES (?, 'default', ?)", (uid, _now()))
        did = cur.lastrowid
        await dbc.commit()
        return uid, did
    finally:
        await _release(dbc)


async def seed_word(dict_id, no, ru="перевод", pos="noun", level="A1"):
    """Добавить слово в пул и в словарь. Вернуть pool_id, dw_id."""
    dbc = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
        cur = await dbc.execute("INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)", (no, data, level, _now()))
        pid = cur.lastrowid
        cur = await dbc.execute("INSERT INTO dict_words (dict_id,pool_id,created_at) VALUES (?,?,?)", (dict_id, pid, _now()))
        dwid = cur.lastrowid
        await dbc.commit()
        return pid, dwid
    finally:
        await _release(dbc)


# доступно тестам как plain helpers
pytest.seed_user = seed_user
pytest.seed_word = seed_word
