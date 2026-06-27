"""Частотность слов (Zipf): градации для UI + выборки по частоте + очередь простановки.
Самостоятельный модуль (зависит только от core); реэкспорт в конце pool.py.
"""
import json
from .core import _conn, _release


def freq_band(z):
    """Zipf-частотность → ключ-градация для UI (подписи во фронте по языкам)."""
    if z is None:
        return None
    if z >= 6:
        return "very_common"
    if z >= 5:
        return "common"
    if z >= 4:
        return "frequent"
    if z >= 3:
        return "occasional"
    if z >= 2:
        return "rare"
    return "very_rare"


async def pool_by_freq(limit: int = 80, level: str = None):
    """Слова пула по убыванию частотности (самые употребимые сначала; freq IS NULL — в хвост).
    [{pool_id, norwegian, translate, part_of_speech, freq}]. Фильтр по уровню CEFR."""
    conds, params = ["data IS NOT NULL", "COALESCE(learn_excluded, 0) = 0"], []
    if level:
        conds.append("level = ?"); params.append(level)
    sql = (f"SELECT id, norwegian, data, freq FROM word_pool WHERE {' AND '.join(conds)} "
           "ORDER BY freq IS NULL, freq DESC LIMIT ?")
    params.append(limit)
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            out = []
            for r in await cur.fetchall():
                try:
                    d = json.loads(r["data"]) or {}
                except Exception:
                    d = {}
                tr = d.get("translate", {}) or {}
                if tr:
                    out.append({"pool_id": r["id"], "norwegian": r["norwegian"], "translate": tr,
                                "part_of_speech": d.get("part_of_speech", ""), "freq": r["freq"]})
            return out
    finally:
        await _release(db)


async def pool_by_freq_topics(limit: int, level, topics):
    """Как pool_by_freq, но только слова с любой из тем `topics` (для фокуса Учёбы). По частоте."""
    if not topics:
        return []
    conds, params = ["wp.data IS NOT NULL", "COALESCE(wp.learn_excluded, 0) = 0"], []
    if level:
        conds.append("wp.level = ?"); params.append(level)
    marks = ",".join("?" for _ in topics)
    sql = (f"SELECT DISTINCT wp.id, wp.norwegian, wp.data, wp.freq FROM word_pool wp "
           f"JOIN word_topics wt ON wt.pool_id = wp.id "
           f"WHERE {' AND '.join(conds)} AND wt.topic IN ({marks}) "
           f"ORDER BY wp.freq IS NULL, wp.freq DESC LIMIT ?")
    params += list(topics); params.append(limit)
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            out = []
            for r in await cur.fetchall():
                try:
                    d = json.loads(r["data"]) or {}
                except Exception:
                    d = {}
                tr = d.get("translate", {}) or {}
                if tr:
                    out.append({"pool_id": r["id"], "norwegian": r["norwegian"], "translate": tr,
                                "part_of_speech": d.get("part_of_speech", ""), "freq": r["freq"]})
            return out
    finally:
        await _release(db)


async def freq_pending(limit: int = 200):
    """Слова без частотности (freq IS NULL) — для фонового добора по корпусу. [(id, norwegian)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian FROM word_pool WHERE freq IS NULL LIMIT ?", (limit,)) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def set_pool_freq(pool_id: int, freq: float):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET freq = ? WHERE id = ?", (float(freq), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_freq_bulk(pairs):
    """pairs: [(id, freq)] — пакетная простановка частотности."""
    if not pairs:
        return 0
    db = await _conn()
    try:
        await db.executemany("UPDATE word_pool SET freq = ? WHERE id = ?",
                             [(float(f), pid) for pid, f in pairs])
        await db.commit()
        return len(pairs)
    finally:
        await _release(db)
