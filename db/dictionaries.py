import json
import aiosqlite
from .core import _conn, _release, _now


async def create_dictionary(user_id: int, name: str):
    name = (name or "").strip()
    if not name:
        return {"error": "Empty name"}
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, name, _now()))
        await db.commit()
        return {"id": cur.lastrowid, "name": name}
    except aiosqlite.IntegrityError:
        return {"error": "Dictionary already exists"}
    finally:
        await _release(db)


async def delete_dictionary(user_id: int, dict_id: int):
    db = await _conn()
    try:
        # не удалять последний словарь
        async with db.execute("SELECT COUNT(*) c FROM dictionaries WHERE user_id = ?", (user_id,)) as cur:
            if (await cur.fetchone())["c"] <= 1:
                return {"error": "Cannot delete the last dictionary"}
        await db.execute("DELETE FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def _owns_dict(db, user_id, dict_id):
    async with db.execute("SELECT id FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id)) as cur:
        return (await cur.fetchone()) is not None


async def add_word_to_dict(user_id: int, dict_id: int, pool_id: int):
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, dict_id):
            return {"error": "Not found"}
        try:
            cur = await db.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?, ?, ?)", (dict_id, pool_id, _now()))
            await db.commit()
            return {"id": cur.lastrowid}
        except aiosqlite.IntegrityError:
            # уже есть в этом словаре
            async with db.execute("SELECT id FROM dict_words WHERE dict_id = ? AND pool_id = ?", (dict_id, pool_id)) as cur:
                row = await cur.fetchone()
                return {"id": row["id"] if row else None, "duplicate": True}
    finally:
        await _release(db)


async def delete_dict_word(user_id: int, dw_id: int):
    db = await _conn()
    try:
        await db.execute("""
            DELETE FROM dict_words WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def move_dict_word(user_id: int, dw_id: int, target_dict_id: int):
    """Перенести слово в другой словарь пользователя, сохранив правки и прогресс.
    Если в целевом словаре оно уже есть — просто убираем из исходного."""
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, target_dict_id):
            return {"error": "Not found"}
        async with db.execute("""
            SELECT dw.id, dw.dict_id FROM dict_words dw
            JOIN dictionaries d ON d.id = dw.dict_id
            WHERE dw.id = ? AND d.user_id = ?
        """, (dw_id, user_id)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "Not found"}
        if row["dict_id"] == target_dict_id:
            return {"ok": True, "moved": False}  # уже в этом словаре
        try:
            await db.execute("UPDATE dict_words SET dict_id = ? WHERE id = ?", (target_dict_id, dw_id))
            await db.commit()
            return {"ok": True, "moved": True}
        except aiosqlite.IntegrityError:
            # UNIQUE(dict_id, pool_id): в целевом словаре слово уже есть — убираем дубль из исходного
            await db.execute("DELETE FROM dict_words WHERE id = ?", (dw_id,))
            await db.commit()
            return {"ok": True, "moved": True, "duplicate": True}
    finally:
        await _release(db)


async def set_word_override(user_id: int, dw_id: int, override: dict):
    db = await _conn()
    try:
        await db.execute("""
            UPDATE dict_words SET override = ?
            WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (json.dumps(override, ensure_ascii=False), dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def record_result(user_id: int, dw_id: int, correct: bool):
    db = await _conn()
    try:
        col = "correct" if correct else "incorrect"
        await db.execute(f"""
            UPDATE dict_words SET {col} = {col} + 1
            WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (dw_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def get_dict_word(user_id: int, dw_id: int):
    """Вернуть (dw_id, pool_id, dict_id, description, data, embedding) для слова пользователя."""
    db = await _conn()
    try:
        async with db.execute("""
            SELECT dw.id, dw.pool_id, dw.dict_id, dw.override, wp.norwegian, wp.description, wp.data, wp.embedding
            FROM dict_words dw
            JOIN dictionaries d ON d.id = dw.dict_id
            JOIN word_pool wp ON wp.id = dw.pool_id
            WHERE dw.id = ? AND d.user_id = ?
        """, (dw_id, user_id)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await _release(db)


def _build_word(row):
    base = json.loads(row["data"]) if row["data"] else {}
    if row["override"]:
        ov = json.loads(row["override"])
        base = {**base, **ov}
        if "translate" in ov:
            base["translate"] = {**base.get("translate", {}), **ov["translate"]}
    desc = json.loads(row["description"]) if row["description"] else None
    return {
        "id": row["dw_id"],
        "word": base.get("word"),
        "translate": base.get("translate", {}),
        "part_of_speech": base.get("part_of_speech", ""),
        "description": {"description": desc} if desc else None,
        "descriptionState": "loaded" if desc else "empty",
        "hasTts": bool(row["has_tts"]),
        "gameData": {"correctFirstTry": row["correct"], "incorrectFirstTry": row["incorrect"], "isChoosedToGame": False},
        "techData": {"isSelected": False},
    }


async def get_user_data(user_id: int):
    """Полная структура словарей пользователя в формате, который ждёт фронтенд."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, name FROM dictionaries WHERE user_id = ? ORDER BY created_at, id", (user_id,)) as cur:
            dicts = [dict(r) for r in await cur.fetchall()]
        result = []
        for d in dicts:
            async with db.execute("""
                SELECT dw.id AS dw_id, dw.override, dw.correct, dw.incorrect,
                       wp.data, wp.description, (wp.tts IS NOT NULL) AS has_tts
                FROM dict_words dw
                JOIN word_pool wp ON wp.id = dw.pool_id
                WHERE dw.dict_id = ?
                ORDER BY dw.id
            """, (d["id"],)) as cur:
                words = [_build_word(r) for r in await cur.fetchall()]
            result.append({"id": d["id"], "dictName": d["name"], "words": words})
        async with db.execute("SELECT current_dict FROM users WHERE id = ?", (user_id,)) as cur:
            urow = await cur.fetchone()
        current = urow["current_dict"] if urow else None
        return {"dictList": result, "dictNames": [d["name"] for d in dicts], "currentDict": current}
    finally:
        await _release(db)
