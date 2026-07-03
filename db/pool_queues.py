"""Аксессоры фоновых очередей пула: озвучка (TTS), эмбеддинги (ANN/семантика),
догенерация переводов и восстановление «ё». Это ПИТАТЕЛИ воркеров (autofill/tts/webpush) —
простые SELECT/UPDATE по колонкам word_pool, отдельный слой от интерактивного CRUD/поиска.
Зависит только от .core; реэкспортируется в db.pool (внешние импорты не меняются).
"""
import json
from .core import _conn, _release, normalize_word, vec_upsert


async def get_pool_tts(norwegian: str):
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT mp3 FROM word_tts WHERE word = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["mp3"] if r and r["mp3"] else None
    finally:
        await _release(db)


async def set_pool_tts(norwegian: str, data: bytes):
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("INSERT OR REPLACE INTO word_tts(word, mp3) VALUES (?, ?)", (key, data))
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
    await vec_upsert(pool_id, data)  # держим sqlite-vec индекс в синхроне (поиск по Базе)
    try:                             # держим резидентный кеш эмбеддингов свежим (дистракторы сессии)
        import embcache
        embcache.update_vec(pool_id, data)
    except Exception:
        pass


async def get_pool_embeddings_raw():
    """[(id, embedding_raw)] для миграции форматов."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, embedding FROM word_pool WHERE embedding IS NOT NULL") as cur:
            return [(r["id"], r["embedding"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_embeddings_page(limit: int = 1000, offset: int = 0):
    """[[id, hex(embedding)]] постранично — админ-выгрузка векторов наружу (мало RAM)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, hex(embedding) AS h FROM word_pool WHERE embedding IS NOT NULL ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            return [[r["id"], r["h"]] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_embedding(limit: int = 1):
    """[(id, norwegian)] — записи без вектора. id нужен, чтобы эмбеддинг записать ИМЕННО
    в эту запись: омонимы (один norwegian → несколько записей с разным pos) имеют каждый
    свой вектор, а get_pool_id без pos попал бы в старшую и NULL у нужной не очистился бы."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian FROM word_pool WHERE embedding IS NULL LIMIT ?", (limit,)) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_tts(limit: int = 1):
    db = await _conn()
    try:
        async with db.execute(
                "SELECT DISTINCT norwegian FROM word_pool "
                "WHERE NOT EXISTS(SELECT 1 FROM word_tts t WHERE t.word = word_pool.norwegian) "
                "LIMIT ?", (limit,)) as cur:
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


async def sem_embed_pending(limit: int = 20):
    """Слова, у которых эмбеддинг ещё не пересчитан по смыслу (emb_sem = 0).
    Возвращает [(id, data_dict)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, data FROM word_pool WHERE COALESCE(emb_sem, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_sem_embed(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET emb_sem = 1 WHERE id = ?", (pool_id,))
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


async def yo_pending(limit: int = 40):
    """Слова, чей русский перевод ещё не проверен на букву «ё» (yo_done = 0).
    Возвращает [(id, data_dict)] — русские переводы берём из data.translate.ru."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, data FROM word_pool WHERE COALESCE(yo_done, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_yo_done(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET yo_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)
