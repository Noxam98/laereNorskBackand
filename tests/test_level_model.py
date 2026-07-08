"""Кумулятивная модель уровней — фикс заморозки currentLevel и запертости словаря плейсмент-тира.

Раньше: by_level[lv].done = (mastered ПО CEFR-ТЕГУ) >= (кумулятивный target) → computed навсегда A1,
currentLevel = start_level статически; suggest фильтровал уровень ТОЧНЫМ равенством → placed-юзер
получал только слова своего тира, а после их исчерпания сессии сохли при нетронутом пуле.
Теперь: уровень по ОБЩЕМУ числу выученных (LEVEL_TARGETS кумулятивны); suggest тянет уровни ≤ current
с fallthrough на любой уровень при нехватке."""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning import LEVEL_TARGETS
from db.learning_suggest import _level_from_total, estimate_level, suggest_words
from db.placement import set_start_level
from tests.conftest import seed_user


# ── чистая карта уровня по кумулятиву ─────────────────────────────────────────
def test_level_from_total_cumulative():
    assert _level_from_total(0) == "A1"
    assert _level_from_total(LEVEL_TARGETS["A1"]) == "A1"       # 250 → достигнут A1
    assert _level_from_total(LEVEL_TARGETS["A2"]) == "A2"       # 500 → A2
    assert _level_from_total(LEVEL_TARGETS["B1"]) == "B1"       # 1000 → B1
    assert _level_from_total(1006) == "B1"                      # между B1 и B2
    assert _level_from_total(LEVEL_TARGETS["C2"]) == "C2"
    # флор плейсмента: слов мало, но placed выше — не опускаем; и не понижаем достигнутый
    assert _level_from_total(50, "B1") == "B1"
    assert _level_from_total(1006, "A1") == "B1"


async def _add_pool(no, level, pos="noun"):
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": ["x"]}, "part_of_speech": pos})
        cur = await db.execute("INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
                               (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _mark_mastered(uid, pids):
    db = await _conn()
    try:
        await db.executemany("INSERT INTO user_words (user_id,pool_id,mastered,created_at) VALUES (?,?,1,?)",
                             [(uid, p, _now()) for p in pids])
        await db.commit()
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_estimate_level_cumulative_and_floor(fresh_db):
    uid, _ = await seed_user()                    # start_level A1 по умолчанию
    assert await estimate_level(uid) == "A1"
    pids = [await _add_pool(f"w{i}", "A1") for i in range(5)]
    await _mark_mastered(uid, pids)
    assert await estimate_level(uid) == "A1"      # 5 выученных — мало для A2 (500)
    await set_start_level(uid, "B1")
    assert await estimate_level(uid) == "B1"      # флор плейсмента держит B1 при 5 словах


@pytest.mark.asyncio
async def test_suggest_cumulative_not_confined(fresh_db):
    """placed-B1: suggest тянет ВСЕ уровни ≤ B1 (не только точный тир) и не выше."""
    uid, _ = await seed_user()
    await set_start_level(uid, "B1")
    a1 = await _add_pool("aone", "A1"); a2 = await _add_pool("atwo", "A2")
    b1 = await _add_pool("bone", "B1"); c1 = await _add_pool("cone", "C1")
    res = await suggest_words(uid, count=3, level="B1")        # ровно 3 доступно ≤B1 → без fallthrough
    added = {w["pool_id"] for w in res["words"]}
    assert added == {a1, a2, b1}                              # кумулятивно ≤B1, C1 не тянем
    assert c1 not in added


@pytest.mark.asyncio
async def test_suggest_fallthrough_never_dry(fresh_db):
    """Уровни ≤ текущего исчерпаны, но в пуле есть слово выше — сессия НЕ сохнет (fallthrough)."""
    uid, _ = await seed_user()                    # A1
    c1 = await _add_pool("solo", "C1")            # в пуле только слово выше уровня
    res = await suggest_words(uid, count=5, level="A1")
    assert c1 in {w["pool_id"] for w in res["words"]}
