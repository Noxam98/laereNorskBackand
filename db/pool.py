import json
from .core import _conn, _release, _now, normalize_word, vec_upsert, vec_delete, vec_nearest_rows


async def get_or_create_pool(norwegian: str, data: dict):
    """Вернуть id записи пула для норвежского слова, создав её при необходимости."""
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            row = await cur.fetchone()
            if row:
                return row["id"]
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(data, ensure_ascii=False), _now()),
        )
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def get_pool_id(norwegian: str):
    """id записи пула по норвежскому слову (без создания)."""
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["id"] if r else None
    finally:
        await _release(db)


async def get_pool_by_id(pool_id: int):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "data": json.loads(row["data"]),
                "description": json.loads(row["description"]) if row["description"] else None,
                "embedding": row["embedding"],  # сырые байты
            }
    finally:
        await _release(db)


async def get_pool_tts(norwegian: str):
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT tts FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["tts"] if r and r["tts"] else None
    finally:
        await _release(db)


async def set_pool_tts(norwegian: str, data: bytes):
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET tts = ? WHERE norwegian = ?", (data, key))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_embedding(pool_id: int, data):
    """data — бинарное представление вектора (float16 bytes)."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET embedding = ? WHERE id = ?", (data, pool_id))
        await db.commit()
    finally:
        await _release(db)
    await vec_upsert(pool_id, data)  # держим ANN-индекс в синхроне


async def get_pool_embeddings_raw():
    """[(id, embedding_raw)] для миграции форматов."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, embedding FROM word_pool WHERE embedding IS NOT NULL") as cur:
            return [(r["id"], r["embedding"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_embedding(limit: int = 1):
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool WHERE embedding IS NULL LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_tts(limit: int = 1):
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool WHERE tts IS NULL LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def translate_pending(limit: int = 10):
    """Слова без отметки translate_done — кандидаты на догенерацию переводов.
    Возвращает [(id, norwegian, data_dict)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data FROM word_pool WHERE COALESCE(translate_done, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], r["norwegian"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_translate_done(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET translate_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def update_pool_translate(pool_id: int, translate: dict):
    """Записать обновлённый словарь translate в data слова; сбросить tts_tr_done,
    чтобы фон озвучил новые языки."""
    db = await _conn()
    try:
        async with db.execute("SELECT data FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return
        data = json.loads(row["data"])
        data["translate"] = translate
        await db.execute(
            "UPDATE word_pool SET data = ?, tts_tr_done = 0 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), pool_id),
        )
        await db.commit()
    finally:
        await _release(db)


async def tr_tts_pending(limit: int = 5):
    """Слова, у которых озвучка переводов ещё не сгенерирована (tts_tr_done = 0).
    Возвращает [(id, data_dict)] — переводы берём из data.translate."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, data FROM word_pool WHERE COALESCE(tts_tr_done, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_tr_tts_done(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET tts_tr_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def get_pool_sample(limit: int = 120):
    """Случайная выборка норвежских слов из пула (исключения для генерации по теме)."""
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool ORDER BY RANDOM() LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_letter(letter: str, limit: int = 120):
    """Норвежские слова пула, начинающиеся на букву (исключения для генерации по букве)."""
    key = normalize_word(letter)
    if not key:
        return []
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian FROM word_pool WHERE norwegian LIKE ? ORDER BY RANDOM() LIMIT ?",
            (key + "%", limit),
        ) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def set_pool_meta(pool_id: int, level: str = None, topics=None):
    """Записать уровень + теги-темы слова (мульти-тег). topics — список ключей."""
    db = await _conn()
    try:
        if level is not None:
            await db.execute("UPDATE word_pool SET level = ? WHERE id = ?", (level, pool_id))
        if topics is not None:
            await db.execute("DELETE FROM word_topics WHERE pool_id = ?", (pool_id,))
            rows = [(pool_id, t) for t in topics if t]
            if rows:
                await db.executemany("INSERT OR IGNORE INTO word_topics (pool_id, topic) VALUES (?, ?)", rows)
        await db.commit()
    finally:
        await _release(db)


async def pool_missing_description(limit: int = 1):
    """Слова без описания — для фоновой догенерации."""
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool WHERE description IS NULL LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_meta(limit: int = 50):
    """Слова без уровня (level IS NULL) + их переводы — для пакетной классификации."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian, data FROM word_pool WHERE level IS NULL LIMIT ?", (limit,)
        ) as cur:
            out = []
            for r in await cur.fetchall():
                d = json.loads(r["data"]) if r["data"] else {}
                out.append({"word": r["norwegian"], "translate": d.get("translate", {})})
            return out
    finally:
        await _release(db)


async def get_pool_stats():
    """Сводка по пулу: всего и сколько с эмбеддингом/озвучкой/описанием/уровнем."""
    db = await _conn()
    try:
        async def one(sql):
            async with db.execute(sql) as cur:
                return (await cur.fetchone())[0]
        return {
            "total": await one("SELECT COUNT(*) FROM word_pool"),
            "embedding": await one("SELECT COUNT(*) FROM word_pool WHERE embedding IS NOT NULL"),
            "tts": await one("SELECT COUNT(*) FROM word_pool WHERE tts IS NOT NULL"),
            "description": await one("SELECT COUNT(*) FROM word_pool WHERE description IS NOT NULL"),
            "classified": await one("SELECT COUNT(*) FROM word_pool WHERE level IS NOT NULL"),
        }
    finally:
        await _release(db)


async def get_pool_topics_counts():
    """[{topic, count}] по непустым темам (для фильтра)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT topic, COUNT(*) c FROM word_topics GROUP BY topic ORDER BY c DESC"
        ) as cur:
            return [{"topic": r["topic"], "count": r["c"]} for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_level_counts():
    """[{level, count}] по проставленным уровням."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT level, COUNT(*) c FROM word_pool WHERE level IS NOT NULL GROUP BY level"
        ) as cur:
            return [{"level": r["level"], "count": r["c"]} for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_candidates():
    """Все слова пула (id, norwegian, data, embedding) — для подбора дистракторов."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian, data, embedding FROM word_pool") as cur:
            rows = await cur.fetchall()
            out = []
            for r in rows:
                out.append({
                    "id": r["id"],
                    "norwegian": r["norwegian"],
                    "data": json.loads(r["data"]) if r["data"] else {},
                    "embedding": r["embedding"],  # сырые байты (декодирует llm)
                })
            return out
    finally:
        await _release(db)


async def set_pool_description(pool_id: int, description: dict):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET description = ? WHERE id = ?", (json.dumps(description, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def delete_pool_word(norwegian: str):
    """Полностью удалить слово из общего пула (каскадом у всех) + почистить кэш запросов."""
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    pid = None
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            pid = r["id"] if r else None
        await db.execute("DELETE FROM word_pool WHERE norwegian = ?", (key,))
        await db.execute(
            "DELETE FROM query_cache WHERE query LIKE ? OR response LIKE ? OR response LIKE ?",
            (f"%{key}%", f"%{key}%", f"%{norwegian}%"),
        )
        await db.commit()
    finally:
        await _release(db)
    await vec_delete(pid)  # убрать из ANN-индекса


_POOL_SORTS = {
    "alpha": "norwegian",
    "added": "created_at, id",
    # уровень: A1<…<C2, непроставленные в конец; добор по алфавиту
    "level": "CASE level WHEN 'A1' THEN 1 WHEN 'A2' THEN 2 WHEN 'B1' THEN 3 "
             "WHEN 'B2' THEN 4 WHEN 'C1' THEN 5 WHEN 'C2' THEN 6 ELSE 99 END, norwegian",
}


async def get_pool_list(limit: int = 60, offset: int = 0, q: str = None,
                        topics=None, level: str = None, sort: str = "alpha", order: str = "asc"):
    """Список слов общего пула: поиск по всем языкам, фильтры тема/уровень,
    сортировка и постраничная пагинация. Возвращает {total, words[]}."""
    conds, params = [], []
    key = normalize_word(q) if q else None
    if key:
        conds.append("(norwegian LIKE ? OR data LIKE ?)")
        params += [f"%{key}%", f"%{key}%"]
    if topics:
        marks = ",".join("?" for _ in topics)
        conds.append(f"id IN (SELECT pool_id FROM word_topics WHERE topic IN ({marks}))")
        params += list(topics)
    if level:
        conds.append("level = ?")
        params.append(level)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    order_by = _POOL_SORTS.get(sort, _POOL_SORTS["alpha"])
    direction = "DESC" if str(order).lower() == "desc" else "ASC"
    # направление применяем к первому ключу сортировки, добор оставляем по возрастанию
    if "," in order_by:
        first, rest = order_by.split(",", 1)
        order_sql = f"{first} {direction},{rest}"
    else:
        order_sql = f"{order_by} {direction}"

    db = await _conn()
    try:
        async with db.execute(f"SELECT COUNT(*) c FROM word_pool {where}", params) as cur:
            total = (await cur.fetchone())["c"]
        async with db.execute(
            f"SELECT id, norwegian, data, level, (tts IS NOT NULL) AS has_tts "
            f"FROM word_pool {where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        # темы страницы — одним запросом (без N+1)
        ids = [r["id"] for r in rows]
        topic_map = {}
        if ids:
            marks = ",".join("?" for _ in ids)
            async with db.execute(
                f"SELECT pool_id, topic FROM word_topics WHERE pool_id IN ({marks})", ids
            ) as cur:
                for tr in await cur.fetchall():
                    topic_map.setdefault(tr["pool_id"], []).append(tr["topic"])
        words = []
        for r in rows:
            d = json.loads(r["data"]) if r["data"] else {}
            words.append({
                "word": r["norwegian"], "translate": d.get("translate", {}),
                "part_of_speech": d.get("part_of_speech", ""),
                "level": r["level"], "topics": topic_map.get(r["id"], []),
                "hasTts": bool(r["has_tts"]),
            })
        return {"total": total, "words": words}
    finally:
        await _release(db)


async def get_pool_ids(q: str = None, topics=None, level: str = None):
    """Все id слов пула, подходящих под фильтр (тот же WHERE, что и get_pool_list) — без пагинации."""
    conds, params = [], []
    key = normalize_word(q) if q else None
    if key:
        conds.append("(norwegian LIKE ? OR data LIKE ?)")
        params += [f"%{key}%", f"%{key}%"]
    if topics:
        marks = ",".join("?" for _ in topics)
        conds.append(f"id IN (SELECT pool_id FROM word_topics WHERE topic IN ({marks}))")
        params += list(topics)
    if level:
        conds.append("level = ?")
        params.append(level)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    db = await _conn()
    try:
        async with db.execute(f"SELECT id FROM word_pool {where}", params) as cur:
            return [r["id"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def search_pool(prefix: str, limit: int = 10):
    """Автокомплит по общему пулу: по норвежскому слову И по переводам (любой язык).
    Норвежские совпадения по префиксу — выше."""
    key = normalize_word(prefix)
    if not key:
        return []
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian, data FROM word_pool "
            "WHERE norwegian LIKE ? OR data LIKE ? "
            "ORDER BY (CASE WHEN norwegian LIKE ? THEN 0 ELSE 1 END), norwegian LIMIT ?",
            (key + "%", "%" + key + "%", key + "%", limit),
        ) as cur:
            rows = await cur.fetchall()
            out = []
            for r in rows:
                d = json.loads(r["data"]) if r["data"] else {}
                out.append({"word": r["norwegian"], "translate": d.get("translate", {}), "part_of_speech": d.get("part_of_speech", "")})
            return out
    finally:
        await _release(db)
