"""IO обратного индекса частей композитов (таблица word_pool_compounds).

Наполнение из ordbank (compound_index_loop) + загрузка в чистую session/compounds.
Логики подбора здесь НЕТ — только доступ к БД и лёгкий TTL-кеш (индекс меняется редко:
лишь новые пул-слова). Направление импорта db→session допустимо (session чистый).
"""
import time

from session import compounds as _comp
from .core import _conn, _release

_CACHE = {"at": 0.0, "index": None, "feat": None, "learnable": None}
_TTL = 300.0   # сек: индекс редкоизменчив, лишний раз не бьём БД


async def pool_batch_after(after_id: int, limit: int = 300):
    """Пул-слова с id > after_id (курсор compound_index_loop) → [(id, norwegian)]."""
    db = await _conn()
    try:
        async with db.execute(
                "SELECT id, norwegian FROM word_pool WHERE id > ? ORDER BY id LIMIT ?",
                (after_id, limit)) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def set_pool_compounds(rows):
    """rows: [(pool_id, norwegian, forledd, etterledd)] → в индекс (идемпотентно)."""
    if not rows:
        return
    db = await _conn()
    try:
        await db.executemany("INSERT OR REPLACE INTO word_pool_compounds VALUES (?,?,?,?)", rows)
        await db.commit()
    finally:
        await _release(db)
    _CACHE["index"] = None   # инвалидируем кеш


async def load_index():
    """(index, feat, learnable) с TTL-кешем.
    index — [{pool_id,no,forledd,etterledd,freq}] всех пул-композитов;
    feat — предвычисленные признаки (session.compounds.build_features);
    learnable — множество частей, которые сами являются словами пула (для предохранителя)."""
    now = time.monotonic()
    if _CACHE["index"] is not None and now - _CACHE["at"] < _TTL:
        return _CACHE["index"], _CACHE["feat"], _CACHE["learnable"]
    db = await _conn()
    try:
        async with db.execute(
                "SELECT c.pool_id, c.norwegian, c.forledd, c.etterledd, COALESCE(w.freq,0) AS freq "
                "FROM word_pool_compounds c JOIN word_pool w ON w.id = c.pool_id") as cur:
            index = [{"pool_id": r["pool_id"], "no": r["norwegian"], "forledd": r["forledd"],
                      "etterledd": r["etterledd"], "freq": r["freq"]} for r in await cur.fetchall()]
        async with db.execute(
                "SELECT DISTINCT p FROM ("
                "  SELECT forledd AS p FROM word_pool_compounds"
                "  UNION SELECT etterledd FROM word_pool_compounds) parts "
                "WHERE p IN (SELECT norwegian FROM word_pool)") as cur:
            learnable = {r["p"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    feat = _comp.build_features(index)
    _CACHE.update(at=now, index=index, feat=feat, learnable=learnable)
    return index, feat, learnable


async def mastered_set(db, user_id):
    """Множество norwegian, доведённых юзером до mastered (для рычага/eligibility)."""
    async with db.execute(
            "SELECT w.norwegian FROM user_words uw JOIN word_pool w ON w.id = uw.pool_id "
            "WHERE uw.user_id = ? AND uw.mastered = 1", (user_id,)) as cur:
        return {r["norwegian"] for r in await cur.fetchall()}


async def compounds_unlocked_by(db, user_id, norwegian):
    """Сколько составных слов НОВО открылось тем, что выучили ИМЕННО это слово-основу: оно —
    часть композита, ВТОРАЯ часть уже mastered, а самого композита у юзера ещё нет. Для
    маленького празднования в apply_result — тем же соединением (в транзакции)."""
    key = (norwegian or "").strip().lower()
    if not key:
        return 0
    async with db.execute(
            "SELECT pool_id, forledd, etterledd FROM word_pool_compounds "
            "WHERE forledd = ? OR etterledd = ?", (key, key)) as cur:
        rows = await cur.fetchall()
    if not rows:
        return 0
    mastered = await mastered_set(db, user_id)
    async with db.execute(
            "SELECT dw.pool_id FROM dict_words dw JOIN dictionaries d ON d.id = dw.dict_id "
            "WHERE d.user_id = ?", (user_id,)) as cur:
        have = {r["pool_id"] for r in await cur.fetchall()}
    n = 0
    for r in rows:
        other = r["etterledd"] if r["forledd"] == key else r["forledd"]
        if other in mastered and r["pool_id"] not in have:
            n += 1
    return n
