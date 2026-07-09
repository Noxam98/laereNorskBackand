"""Смещение подбора новых слов к УРОВНЮ юзера: placed-B1 не заваливается частотными A1-словами
(его потолок), а получает в основном слова своего уровня + ≤⅓ ниже-уровневого «добора пробелов»."""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning_suggest import suggest_words
from db.placement import set_start_level


async def _ins(no, level, freq):
    data = json.dumps({"translate": {"ru": [no + "_ru"]}, "part_of_speech": "noun"})
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, level, freq, approved, created_at) "
            "VALUES (?,?,?,?,1,?)", (no, data, level, freq, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _mark(uid, pid, col):
    db = await _conn()
    try:
        await db.execute("INSERT OR IGNORE INTO user_words (user_id,pool_id,created_at) VALUES (?,?,?)", (uid, pid, _now()))
        await db.execute(f"UPDATE user_words SET {col}=1 WHERE user_id=? AND pool_id=?", (uid, pid))
        await db.commit()
    finally:
        await _release(db)


async def test_known_and_mastered_not_resuggested(fresh_db):
    """known/mastered — жёсткий фильтр: suggest_words не подсыпает их заново (даже если нет dict_words)."""
    uid, did = await pytest.seed_user()
    await set_start_level(uid, "B1")
    b1 = [await _ins(f"b1w{i}", "B1", 4.0 - i * 0.01) for i in range(6)]
    await _mark(uid, b1[0], "known")       # «уже знаю» — не в dict_words
    await _mark(uid, b1[1], "mastered")    # выучено
    res = await suggest_words(uid, count=6, level="B1")
    added = {w["pool_id"] for w in res["words"]}
    assert b1[0] not in added, "known-слово не должно пере-предлагаться"
    assert b1[1] not in added, "mastered-слово не должно пере-предлагаться"


async def test_suggest_biases_to_user_level(fresh_db):
    uid, did = await pytest.seed_user()
    await set_start_level(uid, "B1")
    a1 = {await _ins(f"a1w{i}", "A1", 6.0 - i * 0.01) for i in range(10)}   # частотные (пришли бы первыми)
    b1 = {await _ins(f"b1w{i}", "B1", 4.0 - i * 0.01) for i in range(10)}   # своего уровня, пореже

    res = await suggest_words(uid, count=6, level="B1")
    added = {w["pool_id"] for w in res["words"]}
    assert len(added) == 6
    a1_added = len(added & a1)
    b1_added = len(added & b1)
    # без смещения было бы 6×A1 (они частотнее); квота ниже-уровневых = max(1,(6+2)//3)=2
    assert a1_added <= 2, f"ниже-уровневых слишком много: {a1_added}"
    assert b1_added >= 4, f"своего уровня мало: {b1_added}"


async def test_placement_seeds_known_baseline(fresh_db):
    """Сдал B1 → топ-частотный кор A1/A2 помечен known (не переучиваем), свой уровень не трогаем,
    и suggest их больше не предлагает."""
    from db.placement import _seed_known_baseline
    uid, did = await pytest.seed_user()
    a1 = {await _ins(f"a1w{i}", "A1", 6.0 - i * 0.01) for i in range(5)}
    b1 = {await _ins(f"b1w{i}", "B1", 4.0 - i * 0.01) for i in range(5)}
    await _seed_known_baseline(uid, "B1")
    db = await _conn()
    try:
        async with db.execute("SELECT pool_id FROM user_words WHERE user_id=? AND known=1", (uid,)) as cur:
            known = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert a1 <= known, "A1-кор помечен known"
    assert not (b1 & known), "свой уровень (B1) baseline не трогает"
    res = await suggest_words(uid, count=6, level="B1")
    assert not (a1 & {w["pool_id"] for w in res["words"]}), "known A1 не пере-предлагается"


async def test_suggest_includes_stretch_tier(fresh_db):
    """~20% новых — стретч L+1 (растущий край i+1): B1-юзер видит и B2, не только ≤B1."""
    uid, did = await pytest.seed_user()
    await set_start_level(uid, "B1")
    a1 = {await _ins(f"a1w{i}", "A1", 6.0 - i * 0.01) for i in range(10)}
    b1 = {await _ins(f"b1w{i}", "B1", 4.0 - i * 0.01) for i in range(10)}
    b2 = {await _ins(f"b2w{i}", "B2", 3.0 - i * 0.01) for i in range(10)}
    res = await suggest_words(uid, count=6, level="B1")
    added = {w["pool_id"] for w in res["words"]}
    assert len(added & b2) >= 1, "должен быть стретч B2 (L+1)"
    assert len(added & b1) >= 3, "большинство — своего уровня"
    assert len(added & a1) <= 2, "ниже уровня — ограничено"


async def test_suggest_falls_back_when_own_level_scarce(fresh_db):
    """Если слов своего уровня не хватает — квота снимается, добираем ниже-уровневыми (сессия не сохнет)."""
    uid, did = await pytest.seed_user()
    await set_start_level(uid, "B1")
    a1 = {await _ins(f"a1w{i}", "A1", 6.0 - i * 0.01) for i in range(10)}
    b1 = {await _ins(f"b1w{i}", "B1", 4.0 - i * 0.01) for i in range(2)}    # всего 2 своего уровня

    res = await suggest_words(uid, count=6, level="B1")
    added = {w["pool_id"] for w in res["words"]}
    assert len(added) == 6            # добрали до 6 несмотря на дефицит B1
    assert len(added & b1) == 2       # оба B1 взяты
    assert len(added & a1) == 4       # остальное — A1 (квота снята при нехватке)


async def test_beginner_not_restricted(fresh_db):
    """A1-новичка не режем: below_cap==count, ведёт себя как раньше (частотные вперёд)."""
    uid, did = await pytest.seed_user()
    await set_start_level(uid, "A1")
    a1 = {await _ins(f"a1w{i}", "A1", 6.0 - i * 0.01) for i in range(10)}
    res = await suggest_words(uid, count=6, level="A1")
    assert len({w["pool_id"] for w in res["words"]} & a1) == 6
