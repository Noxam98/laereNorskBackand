"""IO обратного индекса частей композитов (таблица word_pool_compounds).

Наполнение из ordbank (compound_index_loop) + загрузка в чистую session/compounds.
Логики подбора здесь НЕТ — только доступ к БД и лёгкий TTL-кеш (индекс меняется редко:
лишь новые пул-слова). Направление импорта db→session допустимо (session чистый).
"""
import asyncio
import time

from session import compounds as _comp
from .core import _conn, _release

_CACHE = {"at": 0.0, "index": None, "feat": None, "learnable": None}
_TTL = 60.0    # сек: индекс редкоизменчив, но learnable зависит от ВСЕХ слов пула (новое слово-часть
               # индекс не трогает) → короткий TTL ограничивает устаревание, чтение дешёвое
_LOCK = asyncio.Lock()   # single-flight на reload (без дубль-чтения)
_GEN = 0                 # поколение данных: инвалидация ++ → reload не перезапишет кэш устаревшим


def invalidate():
    """Сбросить TTL-кеш индекса (звать при изменении word_pool_compounds ИЛИ добавлении пул-слов,
    т.к. learnable/дистракторы зависят от всего пула). Дёшево — следующий load_index перечитает."""
    global _GEN
    _CACHE["index"] = None
    _GEN += 1


async def pool_batch_after(after_id: int, limit: int = 300):
    """Пул-слова с id > after_id (курсор compound_index_loop) → [(id, norwegian, data)].
    data нужна лупу для единого резолвера (db.pool.resolve_compound): без неё он видел бы
    только банк и пропускал LLM-разобранные слова вне ordbank."""
    import json
    db = await _conn()
    try:
        async with db.execute(
                "SELECT id, norwegian, data FROM word_pool WHERE id > ? ORDER BY id LIMIT ?",
                (after_id, limit)) as cur:
            out = []
            for r in await cur.fetchall():
                try:
                    d = json.loads(r["data"]) if r["data"] else {}
                except Exception:
                    d = {}
                out.append((r["id"], r["norwegian"], d))
            return out
    finally:
        await _release(db)


async def set_pool_compounds(rows):
    """rows: [(pool_id, norwegian, forledd, etterledd)] → в индекс (идемпотентно).
    forledd/etterledd нормализуем в lower — сверяются с пулом (pool.norwegian всегда lower)."""
    if not rows:
        return
    rows = [(pid, no, (fl or "").lower(), (el or "").lower()) for pid, no, fl, el in rows]
    db = await _conn()
    try:
        await db.executemany("INSERT OR REPLACE INTO word_pool_compounds VALUES (?,?,?,?)", rows)
        await db.commit()
    finally:
        await _release(db)
    invalidate()   # инвалидируем кеш (с ++поколения — reload не перезапишет свежую инвалидацию)


async def load_index():
    """(index, feat, learnable) с TTL-кешем (single-flight под _LOCK — без дубль-чтения).
    index — [{pool_id,no,forledd,etterledd,freq}] всех пул-композитов;
    feat — предвычисленные признаки (session.compounds.build_features);
    learnable — множество частей, которые сами являются словами пула (для предохранителя).
    forledd/etterledd приводим к lower на чтении — старые дампы клали части с заглавной,
    а pool.norwegian всегда lower (иначе часть не матчилась бы)."""
    now = time.monotonic()
    if _CACHE["index"] is not None and now - _CACHE["at"] < _TTL:
        return _CACHE["index"], _CACHE["feat"], _CACHE["learnable"]
    async with _LOCK:
        # double-check: пока ждали лок, другой вызов мог уже перечитать
        now = time.monotonic()
        if _CACHE["index"] is not None and now - _CACHE["at"] < _TTL:
            return _CACHE["index"], _CACHE["feat"], _CACHE["learnable"]
        gen_before = _GEN
        db = await _conn()
        try:
            # approved=1 ЖЁСТКО и на ЧТЕНИИ (index/learnable глобальны — их видят ВСЕ юзеры):
            # чужой неодобренный композит иначе разблокировался бы в Учёбе у других (suggest_compounds).
            # Фильтруем на чтении, а не при наполнении: аппрув слова включает его без переиндексации.
            async with db.execute(
                    "SELECT c.pool_id, c.norwegian, c.forledd, c.etterledd, COALESCE(w.freq,0) AS freq "
                    "FROM word_pool_compounds c JOIN word_pool w ON w.id = c.pool_id "
                    "WHERE COALESCE(w.approved,1) = 1") as cur:
                index = [{"pool_id": r["pool_id"], "no": r["norwegian"],
                          "forledd": (r["forledd"] or "").lower(),
                          "etterledd": (r["etterledd"] or "").lower(),
                          "freq": r["freq"]} for r in await cur.fetchall()]
            async with db.execute(
                    "SELECT DISTINCT p FROM ("
                    "  SELECT LOWER(c.forledd) AS p FROM word_pool_compounds c "
                    "    JOIN word_pool w ON w.id = c.pool_id WHERE COALESCE(w.approved,1) = 1"
                    "  UNION SELECT LOWER(c.etterledd) FROM word_pool_compounds c "
                    "    JOIN word_pool w ON w.id = c.pool_id WHERE COALESCE(w.approved,1) = 1) parts "
                    "WHERE p IN (SELECT norwegian FROM word_pool WHERE COALESCE(approved,1) = 1)") as cur:
                learnable = {r["p"] for r in await cur.fetchall()}
        finally:
            await _release(db)
        feat = _comp.build_features(index)
        # кэшируем ТОЛЬКО если за время чтения не было инвалидации (иначе вернём свежие данные,
        # но не запишем — следующий вызов перечитает и не залипнет на устаревшем снимке)
        if _GEN == gen_before:
            _CACHE.update(at=time.monotonic(), index=index, feat=feat, learnable=learnable)
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
    # LOWER(): старые дампы клали части с заглавной, key всегда lower (иначе часть не матчит).
    # JOIN + approved=1: чужой неодобренный композит не считаем «открывшимся» (как в load_index).
    async with db.execute(
            "SELECT c.pool_id, LOWER(c.forledd) AS forledd, LOWER(c.etterledd) AS etterledd "
            "FROM word_pool_compounds c JOIN word_pool w ON w.id = c.pool_id "
            "WHERE COALESCE(w.approved,1) = 1 AND (LOWER(c.forledd) = ? OR LOWER(c.etterledd) = ?)",
            (key, key)) as cur:
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
