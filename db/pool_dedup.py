"""Дедуп пула: поиск дублей (ближайшие соседи по vec-индексу) + слияние слов-дублей.
Зависит только от core (+ ленивые numpy/_f16 внутри nearest_other); реэкспорт в конце pool.py.
"""
from .core import _conn, _release


async def dedup_pending(limit: int = 1):
    """Слова в очереди фонового дедупа (dedup_done=0, есть эмбеддинг). Новые — первыми (id DESC).
    Возвращает [(id, norwegian, data, embedding_raw)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data, embedding FROM word_pool "
            "WHERE COALESCE(dedup_done, 0) = 0 AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [(r["id"], r["norwegian"], r["data"], r["embedding"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_dedup(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET dedup_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def pool_usage_count(pool_id: int):
    """Сколько раз слово используется в словарях (популярность)."""
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM dict_words WHERE pool_id = ?", (pool_id,)) as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def nearest_other(pool_id: int, embedding_raw, k: int = 4):
    """Ближайшие соседи по vec-индексу, кроме самого слова: [(id, norwegian, data, distance)].
    distance — косинус-дистанция (cos = 1 - distance). [] если индекс недоступен/пуст."""
    import numpy as np
    from .core import _f16_to_f32_bytes
    q = _f16_to_f32_bytes(embedding_raw, np)
    if q is None:
        return []
    db = await _conn()
    try:
        try:
            async with db.execute(
                "SELECT rowid, distance FROM vec_words WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (q, k + 1),
            ) as cur:
                knn = await cur.fetchall()
        except Exception:
            return []
        out = []
        for r in knn:
            if r["rowid"] == pool_id:
                continue
            async with db.execute("SELECT norwegian, data FROM word_pool WHERE id = ?", (r["rowid"],)) as c2:
                w = await c2.fetchone()
            if w:
                out.append((r["rowid"], w["norwegian"], w["data"], r["distance"]))
        return out
    finally:
        await _release(db)


async def merge_pool_words(winner_id: int, loser_id: int):
    """Слить loser в winner: перепривязать dict_words/user_words на победителя (OR IGNORE при
    UNIQUE-конфликте), удалить остатки и сам loser из word_pool/word_topics/vec_words. True при успехе."""
    if winner_id == loser_id:
        return False
    db = await _conn()
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("UPDATE OR IGNORE dict_words SET pool_id = ? WHERE pool_id = ?", (winner_id, loser_id))
        await db.execute("UPDATE OR IGNORE user_words SET pool_id = ? WHERE pool_id = ?", (winner_id, loser_id))
        await db.execute("DELETE FROM dict_words  WHERE pool_id = ?", (loser_id,))
        await db.execute("DELETE FROM user_words  WHERE pool_id = ?", (loser_id,))
        await db.execute("DELETE FROM word_topics WHERE pool_id = ?", (loser_id,))
        try:
            await db.execute("DELETE FROM vec_words WHERE rowid = ?", (loser_id,))
        except Exception:
            pass
        await db.execute("DELETE FROM word_pool WHERE id = ?", (loser_id,))
        await db.commit()
    finally:
        await _release(db)
    try:                              # убрать loser из резидентного кеша эмбеддингов (дистракторы)
        import embcache
        embcache.remove_vec(loser_id)
    except Exception:
        pass
    return True


async def dedup_progress():
    """(проверено, всего) для отслеживания фонового дедупа."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(*) tot, SUM(CASE WHEN COALESCE(dedup_done,0)=1 THEN 1 ELSE 0 END) done "
            "FROM word_pool WHERE embedding IS NOT NULL"
        ) as cur:
            r = await cur.fetchone()
            return (r["done"] or 0, r["tot"] or 0)
    finally:
        await _release(db)
