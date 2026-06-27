"""Модерация пользовательских слов (личное расширение → общая база) + жалобы «не учить».
Вынесено из pool.py; реэкспортируется в конце pool.py (db/__init__ и импорты не меняются).
"""
from .core import _conn, _release, _now
from .pool import _loads


# ---------------- Модерация пользовательских слов (личное расширение → общая база) ----------------
async def pending_words(limit: int = 300, offset: int = 0):
    """Слова на модерации (approved=0) по всем юзерам — для админа, с автором."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT wp.id, wp.norwegian, wp.data, wp.pos, wp.created_at, wp.created_by, "
            "u.username AS author "
            "FROM word_pool wp LEFT JOIN users u ON u.id = wp.created_by "
            "WHERE COALESCE(wp.approved, 1) = 0 "
            "ORDER BY wp.created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            d = _loads(r["data"])
            out.append({
                "pool_id": r["id"], "word": r["norwegian"],
                "part_of_speech": r["pos"] or d.get("part_of_speech", ""),
                "translate": d.get("translate", {}),
                "level": d.get("level"), "topics": d.get("topics", []),
                "author": r["author"], "author_id": r["created_by"], "created_at": r["created_at"],
            })
        return out
    finally:
        await _release(db)


async def pending_count():
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM word_pool WHERE COALESCE(approved, 1) = 0") as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def set_word_approval(pool_id: int, approved: int):
    """approved: 1 — одобрить (в общую базу), 2 — отклонить (остаётся приватным у автора)."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET approved = ? WHERE id = ?", (int(approved), pool_id))
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "approved": int(approved)}
    finally:
        await _release(db)


# ---------------- Жалобы «не учить» (мусорные слова → модерация → убрать из учёбы / оставить) ----------------
async def report_word(pool_id: int, user_id: int):
    """Пользователь жалуется «не учить». Убираем слово из ЕГО учёбы и решаем судьбу жалобы:
      • learn_excluded=1     → уже убрано из учёбы для всех (status=excluded, тихо);
      • report_dismiss_left>0 → админ ранее решил «оставить» → гасим жалобу (−1, status=dismissed);
      • иначе                 → ставим в очередь админа (reported=1, report_count+1, status=queued)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COALESCE(learn_excluded,0) le, COALESCE(report_dismiss_left,0) dl "
            "FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "not_found"}
        # «Отправить на модерацию» = в персональную СВАЛКУ юзера: учить НЕ будет НАВСЕГДА, независимо
        # от решения модератора. Убираем из его словарей (dict_words) и прогресса (user_words —
        # удаляем, это мусор, не «выучено»), и заносим в user_word_skips, чтобы suggest_words больше
        # никогда не предлагал это слово ЭТОМУ юзеру (даже если модератор слово оставит).
        await db.execute(
            "DELETE FROM dict_words WHERE pool_id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)",
            (pool_id, user_id))
        await db.execute("DELETE FROM user_words WHERE user_id = ? AND pool_id = ?", (user_id, pool_id))
        await db.execute("INSERT OR IGNORE INTO user_word_skips (user_id, pool_id, created_at) VALUES (?,?,?)", (user_id, pool_id, _now()))
        if row["le"]:
            status = "excluded"
        elif row["dl"] > 0:
            await db.execute("UPDATE word_pool SET report_dismiss_left = report_dismiss_left - 1 WHERE id = ?", (pool_id,))
            status = "dismissed"
        else:
            await db.execute("UPDATE word_pool SET reported = 1, report_count = COALESCE(report_count,0) + 1 WHERE id = ?", (pool_id,))
            status = "queued"
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "status": status}
    finally:
        await _release(db)


async def reported_words(limit: int = 300, offset: int = 0):
    """Слова с активными жалобами «не учить» (reported=1) — для админа."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data, pos, level, COALESCE(report_count,0) rc "
            "FROM word_pool WHERE COALESCE(reported,0) = 1 "
            "ORDER BY rc DESC, id DESC LIMIT ? OFFSET ?", (limit, offset)) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            d = _loads(r["data"])
            out.append({
                "pool_id": r["id"], "word": r["norwegian"],
                "part_of_speech": r["pos"] or d.get("part_of_speech", ""),
                "translate": d.get("translate", {}),
                "level": r["level"] or d.get("level"),
                "reports": r["rc"],
            })
        return out
    finally:
        await _release(db)


async def reported_count():
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM word_pool WHERE COALESCE(reported,0) = 1") as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def resolve_report(pool_id: int, action: str):
    """Вердикт админа по жалобе:
      • 'exclude' — убрать из учёбы (learn_excluded=1, снять жалобу);
      • 'keep'    — оставить слово, следующие 5 жалоб гасить автоматически (report_dismiss_left=5)."""
    db = await _conn()
    try:
        if action == "exclude":
            await db.execute(
                "UPDATE word_pool SET learn_excluded = 1, reported = 0, report_count = 0, report_dismiss_left = 0 WHERE id = ?",
                (pool_id,))
        elif action == "keep":
            await db.execute(
                "UPDATE word_pool SET reported = 0, report_count = 0, report_dismiss_left = 5 WHERE id = ?",
                (pool_id,))
        else:
            return {"error": "bad_action"}
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "action": action}
    finally:
        await _release(db)
