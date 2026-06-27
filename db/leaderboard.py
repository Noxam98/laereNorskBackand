"""Рейтинг учеников + дневная активность для хитмапа.
Самостоятельные SQL-запросы поверх соединения — без зависимостей от движка «Учёбы»
(вынесено из learning.py, чтобы тот не разрастался). Реэкспортируется через db и learning.
"""
from .core import _conn, _release


async def get_activity(user_id, days=119):
    """Дневная активность за период (для хитмапа). [{day, answers, correct}]."""
    from datetime import date, timedelta
    cut = (date.today() - timedelta(days=days)).isoformat()
    db = await _conn()
    try:
        async with db.execute("SELECT day, answers, correct FROM user_activity WHERE user_id=? AND day>=? ORDER BY day", (user_id, cut)) as cur:
            return {"days": [dict(r) for r in await cur.fetchall()]}
    finally:
        await _release(db)


async def learning_leaderboard(user_id, period="week", limit=50):
    """Рейтинг учеников. period='week' — очки = верные ответы за текущую неделю (Пн, UTC);
    period='all' — число выученных слов (mastered). Скрываем тех, кто выключил участие
    (game_prefs.leaderboardOptOut). Логин/email не отдаём; имя = display_name или None («Аноним»)."""
    from datetime import datetime, timedelta
    notout = "COALESCE(json_extract(u.game_prefs,'$.leaderboardOptOut'),0)=0"   # по умолчанию участвуют все
    week_start = None
    if period == "all":
        sql = f"""
            SELECT u.id AS uid, u.display_name AS name, u.start_level AS level, COUNT(w.id) AS pts, 0 AS ans
            FROM users u JOIN user_words w ON w.user_id=u.id AND w.mastered=1
            WHERE {notout}
            GROUP BY u.id HAVING pts > 0
            ORDER BY pts DESC, u.id ASC
        """
        params = ()
    else:
        now = datetime.utcnow()
        week_start = (now - timedelta(days=now.weekday())).date().isoformat()   # понедельник этой недели (UTC)
        sql = f"""
            SELECT u.id AS uid, u.display_name AS name, u.start_level AS level,
                   COALESCE(SUM(a.correct),0) AS pts, COALESCE(SUM(a.answers),0) AS ans
            FROM users u JOIN user_activity a ON a.user_id=u.id AND a.day >= ?
            WHERE {notout}
            GROUP BY u.id HAVING pts > 0
            ORDER BY pts DESC, ans ASC, u.id ASC
        """
        params = (week_start,)
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT COALESCE(json_extract(game_prefs,'$.leaderboardOptOut'),0) AS o FROM users WHERE id=?",
            (user_id,)) as cur:
            orow = await cur.fetchone()
    finally:
        await _release(db)
    opted_out = bool(orow and orow["o"])
    top, me = [], None
    for i, r in enumerate(rows):
        entry = {"rank": i + 1, "name": (r["name"] or None), "level": (r["level"] or None),
                 "points": r["pts"], "me": r["uid"] == user_id}
        if entry["me"]:
            me = entry
        if i < limit:
            top.append(entry)
    return {"period": ("all" if period == "all" else "week"), "weekStart": week_start,
            "count": len(rows), "top": top, "me": me, "optedOut": opted_out}
