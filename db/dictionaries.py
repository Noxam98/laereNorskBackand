import json
import aiosqlite
from .core import _conn, _release, _now
from .pool import freq_band


async def get_user_quiz_words(user_id: int, dict_id=None, limit: int = 80):
    """Слова из словарей пользователя для онлайн-игры: [{norwegian, translate, embedding}].
    dict_id=None — из всех словарей, иначе из конкретного. Перевод — с учётом override."""
    conds, params = ["d.user_id = ?"], [user_id]
    if dict_id:
        conds.append("d.id = ?"); params.append(dict_id)
    sql = (f"SELECT wp.norwegian, wp.data, wp.embedding, dw.override "
           f"FROM dict_words dw JOIN dictionaries d ON d.id = dw.dict_id "
           f"JOIN word_pool wp ON wp.id = dw.pool_id "
           f"WHERE {' AND '.join(conds)} ORDER BY RANDOM() LIMIT ?")
    params.append(limit)
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            out = []
            for r in await cur.fetchall():
                base = json.loads(r["data"]) if r["data"] else {}
                tr = dict(base.get("translate", {}) or {})
                if r["override"]:
                    ov = json.loads(r["override"])
                    if ov.get("translate"):
                        tr = {**tr, **ov["translate"]}
                if tr:
                    out.append({"norwegian": r["norwegian"], "translate": tr, "embedding": r["embedding"]})
            return out
    finally:
        await _release(db)


async def create_dictionary(user_id: int, name: str):
    name = (name or "").strip()
    if not name:
        return {"error": "Empty name"}
    db = await _conn()
    try:
        # studying=0: новый личный словарь по умолчанию НЕ в «Учёбе» (включается в «Действиях»).
        cur = await db.execute("INSERT INTO dictionaries (user_id, name, created_at, studying) VALUES (?, ?, ?, 0)", (user_id, name, _now()))
        await db.commit()
        return {"id": cur.lastrowid, "name": name}
    except aiosqlite.IntegrityError:
        return {"error": "Dictionary already exists"}
    finally:
        await _release(db)


HIDDEN_DICT_NAME = "__auto__"   # имя скрытого авто-словаря для «докинуть»/стартового набора


async def get_or_create_hidden_dict(user_id: int) -> int:
    """Получить-или-создать скрытый авто-словарь пользователя (hidden=1, studying=1).
    Сюда «докидываем» новые слова, чтобы не засорять личные словари; в Учёбе он виден
    (studying=1), в «Мой словарь» — нет (hidden=1). Возвращает dict_id."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id FROM dictionaries WHERE user_id = ? AND COALESCE(hidden, 0) = 1 ORDER BY created_at, id LIMIT 1",
            (user_id,)) as cur:
            row = await cur.fetchone()
        if row:
            return row["id"]
        try:
            cur = await db.execute(
                "INSERT INTO dictionaries (user_id, name, created_at, hidden, studying) VALUES (?, ?, ?, 1, 1)",
                (user_id, HIDDEN_DICT_NAME, _now()))
            await db.commit()
            return cur.lastrowid
        except aiosqlite.IntegrityError:
            # имя занято (старый явный словарь с таким именем) — сделаем его скрытым и переиспользуем
            async with db.execute(
                "SELECT id FROM dictionaries WHERE user_id = ? AND name = ?", (user_id, HIDDEN_DICT_NAME)) as cur:
                row = await cur.fetchone()
            if row:
                await db.execute("UPDATE dictionaries SET hidden = 1, studying = 1 WHERE id = ?", (row["id"],))
                await db.commit()
                return row["id"]
            raise
    finally:
        await _release(db)


async def set_dictionary_studying(user_id: int, dict_id: int, studying: bool):
    """Тоггл флага «в обучении» на личном словаре. Скрытый авто-словарь нельзя выключать."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COALESCE(hidden, 0) AS hidden FROM dictionaries WHERE id = ? AND user_id = ?",
            (dict_id, user_id)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "Not found"}
        if row["hidden"]:
            return {"error": "Cannot change studying for the hidden dictionary"}
        await db.execute("UPDATE dictionaries SET studying = ? WHERE id = ? AND user_id = ?",
                         (1 if studying else 0, dict_id, user_id))
        await db.commit()
        return {"id": dict_id, "studying": bool(studying)}
    finally:
        await _release(db)


async def rename_dictionary(user_id: int, dict_id: int, name: str):
    name = (name or "").strip()
    if not name:
        return {"error": "Empty name"}
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id)) as cur:
            if not await cur.fetchone():
                return {"error": "Not found"}
        try:
            await db.execute("UPDATE dictionaries SET name = ? WHERE id = ? AND user_id = ?", (name, dict_id, user_id))
            await db.commit()
        except aiosqlite.IntegrityError:
            return {"error": "Dictionary already exists"}
        return {"id": dict_id, "name": name}
    finally:
        await _release(db)


async def delete_dictionary(user_id: int, dict_id: int):
    db = await _conn()
    try:
        # не удалять последний словарь
        async with db.execute("SELECT COUNT(*) c FROM dictionaries WHERE user_id = ?", (user_id,)) as cur:
            if (await cur.fetchone())["c"] <= 1:
                return {"error": "Cannot delete the last dictionary"}
        # сначала членство (dict_words не каскадятся по FK), затем сам словарь.
        # Прогресс SRS (user_words) НЕ трогаем: слово остаётся выученным, если оно ещё
        # в другом наборе или уже начато (см. _fetch_user_words «общий прогресс»).
        await db.execute("DELETE FROM dict_words WHERE dict_id IN (SELECT id FROM dictionaries WHERE id = ? AND user_id = ?)", (dict_id, user_id))
        await db.execute("DELETE FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def _owns_dict(db, user_id, dict_id):
    async with db.execute("SELECT id FROM dictionaries WHERE id = ? AND user_id = ?", (dict_id, user_id)) as cur:
        return (await cur.fetchone()) is not None


# Личные наборы («sets») вынесены в db/sets_data.py (реэкспорт в конце файла).
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


async def record_result(user_id: int, dw_id: int, correct: bool, mode: str = None, elapsed: float = None, direction: str = None):
    db = await _conn()
    try:
        col = "correct" if correct else "incorrect"
        async with db.execute("""
            SELECT dw.pool_id FROM dict_words dw
            JOIN dictionaries d ON d.id = dw.dict_id
            WHERE dw.id = ? AND d.user_id = ?
        """, (dw_id, user_id)) as cur:
            row = await cur.fetchone()
        pool_id = row["pool_id"] if row else None
        await db.execute(f"""
            UPDATE dict_words SET {col} = {col} + 1
            WHERE id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)
        """, (dw_id, user_id))
        await db.commit()
    finally:
        await _release(db)
    # Игра кормит SRS «Учёбы»: любой ответ обновляет силу/интервал; mode — для серии «без ошибок».
    if pool_id is not None:
        try:
            from .learning import apply_result
            await apply_result(user_id, pool_id, correct, elapsed=elapsed, mode=mode, direction=direction)
        except Exception:
            pass
    return {"ok": True}


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
    forms_raw = row["forms"] if "forms" in row.keys() else None
    return {
        "id": row["dw_id"],
        "pool_id": (row["pool_id"] if "pool_id" in row.keys() else None),
        "word": base.get("word"),
        "translate": base.get("translate", {}),
        "part_of_speech": base.get("part_of_speech", ""),
        "forms": json.loads(forms_raw) if forms_raw else None,
        "description": {"description": desc} if desc else None,
        "descriptionState": "loaded" if desc else "empty",
        "hasTts": bool(row["has_tts"]),
        "freq": (row["freq"] if "freq" in row.keys() else None),
        "freqBand": freq_band(row["freq"] if "freq" in row.keys() else None),
        "gameData": {"correctFirstTry": row["correct"], "incorrectFirstTry": row["incorrect"], "isChoosedToGame": False},
        "techData": {"isSelected": False},
    }


async def get_user_data(user_id: int):
    """Полная структура словарей пользователя в формате, который ждёт фронтенд."""
    db = await _conn()
    try:
        # скрытый авто-словарь (hidden=1) не показываем в «Мой словарь»
        async with db.execute("SELECT id, name, COALESCE(studying, 1) AS studying FROM dictionaries WHERE user_id = ? AND COALESCE(hidden, 0) = 0 ORDER BY created_at, id", (user_id,)) as cur:
            dicts = [dict(r) for r in await cur.fetchall()]
        result = []
        for d in dicts:
            async with db.execute("""
                SELECT dw.id AS dw_id, dw.pool_id, dw.override, dw.correct, dw.incorrect,
                       wp.data, wp.description, wp.forms, wp.freq, EXISTS(SELECT 1 FROM word_tts t WHERE t.word = wp.norwegian) AS has_tts
                FROM dict_words dw
                JOIN word_pool wp ON wp.id = dw.pool_id
                WHERE dw.dict_id = ?
                ORDER BY dw.id
            """, (d["id"],)) as cur:
                words = [_build_word(r) for r in await cur.fetchall()]
            result.append({"id": d["id"], "dictName": d["name"], "studying": bool(d["studying"]), "words": words})
        async with db.execute("SELECT current_dict FROM users WHERE id = ?", (user_id,)) as cur:
            urow = await cur.fetchone()
        current = urow["current_dict"] if urow else None
        return {"dictList": result, "dictNames": [d["name"] for d in dicts], "currentDict": current}
    finally:
        await _release(db)


from .sets_data import (  # noqa: E402,F401
    list_user_sets, add_words_to_set, remove_word_from_set,
    get_set_words, reset_set_ramp, sets_for_words,
)