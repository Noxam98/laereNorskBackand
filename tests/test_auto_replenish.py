"""Авто-добор «Учёбы» (feat/auto-replenish): система ДОСЫПАЕТ новые слова САМА,
когда живой пул (new+learning) иссякает, — без кнопки «Докинуть».

Логика в build_session (db/learning.py): если ворота НЕ закрыты И живой пул < WIP_LIMIT —
suggest_words досыпает кандидатов уровня из «Базы» (word_pool) в скрытый авто-словарь,
после чего сессия пересобирается и становится непустой.

Что проверяем (через публичный build_session, без подгонки под детали реализации):
  (1) пустой живой пул + есть слова «Базы» под уровень → добор сработал: в скрытом авто-словаре
      появились слова, сессия непустая;
  (2) ворота закрыты (несданная пачка >= порога) → добор НЕ срабатывает (новые заблокированы);
  (3) живой пул уже >= WIP_LIMIT (свои новые в включённом словаре) → не досыпает, без переполнения;
  (4) у юзера есть свои новые (включённый словарь studying=1) → «База» не подмешивается.
"""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning import build_session, apply_result, WIP_LIMIT, PACK_FIRST
from tests.conftest import seed_user, seed_word


# ---------------- хелперы ----------------

async def _add_base_word(no, ru="перевод", level="A1"):
    """Слово только в «Базу» (общий word_pool), ни в один словарь юзера. Вернуть pool_id."""
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": "noun"})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
            (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _seed_base(n, prefix="base", level="A1"):
    """Засеять n слов «Базы» под уровень (кандидаты авто-добора)."""
    return [await _add_base_word(f"{prefix}{i}", f"пер{i}", level) for i in range(n)]


async def _hidden_word_count(user_id):
    """Сколько слов лежит в скрытом авто-словаре пользователя."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(*) AS n FROM dict_words "
            "WHERE dict_id IN (SELECT id FROM dictionaries "
            "                  WHERE user_id = ? AND COALESCE(hidden,0) = 1)",
            (user_id,)) as cur:
            return (await cur.fetchone())["n"]
    finally:
        await _release(db)


async def _master(uid, pid):
    """Довести слово до mastered (вся рампа из 4 клеток)."""
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    await apply_result(uid, pid, True, mode="input", direction="int2no")


# ---------------- (1) пустой живой пул + «База» → добор сработал ----------------

@pytest.mark.asyncio
async def test_replenish_when_pool_empty_and_base_available(fresh_db):
    uid, _did = await seed_user()
    # дефолтный словарь пуст; в «Базе» есть кандидаты уровня A1 (estimate_level новичка → A1)
    await _seed_base(12, level="A1")

    assert await _hidden_word_count(uid) == 0   # до сессии ничего не досыпано

    res = await build_session(uid, size=10)

    # система сама досыпала: в скрытый авто-словарь добавились слова, сессия непустая
    assert await _hidden_word_count(uid) >= 1
    assert len(res["words"]) >= 1


# ---------------- (2) ворота закрыты → НЕ досыпает ----------------

@pytest.mark.asyncio
async def test_no_replenish_when_gate_closed(fresh_db):
    uid, did = await seed_user()
    # несданная пачка достигла первого порога → ворота закрыты, приток новых заблокирован
    for i in range(PACK_FIRST):
        pid, _ = await seed_word(did, f"pack{i}", f"пак{i}")
        await _master(uid, pid)
    # в «Базе» есть кандидаты — но при закрытых воротах их подмешивать нельзя
    await _seed_base(12, level="A1")

    res = await build_session(uid, size=10)

    # добор не сработал: скрытый авто-словарь пуст
    assert await _hidden_word_count(uid) == 0
    # ни одно слово сессии не из «Базы» (их вообще не добавляли в словари юзера)
    base_in_session = [w for w in res["words"] if w["no"].startswith("base")]
    assert base_in_session == []


# ---------------- (3) живой пул уже >= WIP_LIMIT → не досыпает ----------------

@pytest.mark.asyncio
async def test_no_replenish_when_pool_at_wip_limit(fresh_db):
    uid, did = await seed_user()
    # ровно WIP_LIMIT новых слов у юзера (включённый словарь) → живой пул == WIP_LIMIT
    for i in range(WIP_LIMIT):
        await seed_word(did, f"own{i}", f"свой{i}")
    # в «Базе» тоже есть кандидаты — но добор не нужен, пул полон
    await _seed_base(12, level="A1")

    res = await build_session(uid, size=10)

    # ничего не досыпано (без переполнения)
    assert await _hidden_word_count(uid) == 0
    # сессия собрана из своих слов, не из «Базы»
    assert len(res["words"]) >= 1
    assert all(w["no"].startswith("own") for w in res["words"])


# ---------------- (4) есть свои новые → «База» не подмешивается ----------------

@pytest.mark.asyncio
async def test_no_replenish_when_user_has_own_new_words(fresh_db):
    uid, did = await seed_user()
    # у юзера полный пул своих новых во включённом словаре (studying=1 по умолчанию)
    own = [(await seed_word(did, f"mine{i}", f"моё{i}"))[0] for i in range(WIP_LIMIT)]
    own_set = set(own)
    # «База» под тот же уровень существует, но не должна попасть в Учёбу
    await _seed_base(12, level="A1")

    res = await build_session(uid, size=10)

    # добор не сработал — «База» не подмешана в скрытый авто-словарь
    assert await _hidden_word_count(uid) == 0
    # все слова сессии — свои; ни одного из «Базы»
    assert len(res["words"]) >= 1
    assert all(w["pool_id"] in own_set for w in res["words"])
    assert all(w["no"].startswith("mine") for w in res["words"])
