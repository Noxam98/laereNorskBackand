import pytest
from db.core import _conn, _release, _now
from db import learning_leaderboard


async def _seed_activity(uid, day, answers, correct):
    db = await _conn()
    try:
        await db.execute("INSERT OR REPLACE INTO user_activity (user_id,day,answers,correct) VALUES (?,?,?,?)",
                         (uid, day, answers, correct))
        await db.commit()
    finally:
        await _release(db)


async def _set_opt_out(uid, val):
    import json
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_prefs=? WHERE id=?", (json.dumps({"leaderboardOptOut": val}), uid))
        await db.commit()
    finally:
        await _release(db)


async def _set_mastered(uid, did, no):
    pid, _dw = await pytest.seed_word(did, no)
    db = await _conn()
    try:
        await db.execute("INSERT INTO user_words (user_id,pool_id,mastered,created_at) VALUES (?,?,1,?)",
                         (uid, pid, _now()))
        await db.commit()
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_weekly_leaderboard_ranks_and_me(fresh_db):
    from datetime import datetime
    today = datetime.utcnow().date().isoformat()
    u1, d1 = await pytest.seed_user("alice")
    u2, d2 = await pytest.seed_user("bob")
    u3, d3 = await pytest.seed_user("carol")
    # имя только у alice
    db = await _conn()
    try:
        await db.execute("UPDATE users SET display_name=? WHERE id=?", ("Alice", u1)); await db.commit()
    finally:
        await _release(db)
    await _seed_activity(u1, today, 10, 8)
    await _seed_activity(u2, today, 20, 15)
    await _seed_activity(u3, today, 5, 5)
    res = await learning_leaderboard(u1, period="week")
    assert res["period"] == "week" and res["count"] == 3
    names = [(e["rank"], e["points"], e["me"], e["name"]) for e in res["top"]]
    assert names[0] == (1, 15, False, None)      # bob top, без имени → None
    assert names[1] == (2, 8, True, "Alice")     # alice вторая, помечена me
    assert names[2] == (3, 5, False, None)
    assert res["me"]["rank"] == 2 and res["me"]["points"] == 8
    # очки не попавшие в неделю не учитываются
    res2 = await learning_leaderboard(u3, period="week", limit=2)
    assert len(res2["top"]) == 2 and res2["me"]["rank"] == 3   # carol вне топ-2, но me даёт её ранг


@pytest.mark.asyncio
async def test_opt_out_hidden(fresh_db):
    from datetime import datetime
    today = datetime.utcnow().date().isoformat()
    u1, d1 = await pytest.seed_user("a")
    u2, d2 = await pytest.seed_user("b")
    await _seed_activity(u1, today, 10, 9)
    await _seed_activity(u2, today, 10, 7)
    await _set_opt_out(u1, True)
    res = await learning_leaderboard(u1, period="week")
    assert res["optedOut"] is True
    assert res["count"] == 1                       # только u2 виден
    assert res["me"] is None                        # сам скрыт
    assert res["top"][0]["points"] == 7


@pytest.mark.asyncio
async def test_alltime_by_mastered(fresh_db):
    u1, d1 = await pytest.seed_user("a")
    u2, d2 = await pytest.seed_user("b")
    await _set_mastered(u1, d1, "katt")
    await _set_mastered(u1, d1, "hund")
    await _set_mastered(u2, d2, "hus")
    res = await learning_leaderboard(u1, period="all")
    assert res["period"] == "all" and res["weekStart"] is None
    assert res["top"][0]["points"] == 2 and res["top"][0]["me"] is True
    assert res["count"] == 2
