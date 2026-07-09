"""Личные наборы для изучения («sets» = словари пользователя с hidden=0).
CRUD набора + слова в нём + сборка слов набора с рампой (ramp) + сброс рампы.
Вынесено из dictionaries.py; реэкспорт в конце dictionaries.py (db/__init__ не меняется).
"""
import json
import aiosqlite
from .core import _conn, _release, _now
from .dictionaries import _owns_dict


# ---------------- личные наборы для изучения («sets» = словари с hidden=0) ----------------
# Набор = обычный словарь пользователя (hidden=0). Прогресс SRS общий (user_words по pool_id),
# набор — лишь членство (dict_words) + флаг studying («питать ли ежедневную умную сессию»).

async def list_user_sets(user_id: int):
    """Личные наборы (hidden=0) с числом слов и флагом studying. Авто-словарь (hidden=1) скрыт.
    Рудимент: пустой системный словарь 'default' (создаётся при регистрации, но в него ничего
    не кладётся — слова идут в скрытый авто-словарь) НЕ показываем как набор. Если у легаси-юзера
    в 'default' есть слова — показываем (чтобы не прятать его слова)."""
    db = await _conn()
    try:
        async with db.execute("""
            SELECT d.id, d.name, COALESCE(d.studying, 0) AS studying,
                   (SELECT COUNT(*) FROM dict_words dw WHERE dw.dict_id = d.id) AS cnt
            FROM dictionaries d
            WHERE d.user_id = ? AND COALESCE(d.hidden, 0) = 0
            ORDER BY d.created_at, d.id
        """, (user_id,)) as cur:
            return [{"id": r["id"], "name": r["name"], "studying": bool(r["studying"]), "count": r["cnt"]}
                    for r in await cur.fetchall()
                    if not (r["name"] == "default" and r["cnt"] == 0)]
    finally:
        await _release(db)


async def add_words_to_set(user_id: int, set_id: int, pool_ids):
    """Массово добавить слова пула в набор (дубли игнорируем). Возвращает {added}."""
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, set_id):
            return {"error": "Not found"}
        added = 0
        for pid in [int(p) for p in (pool_ids or []) if p]:
            try:
                await db.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?, ?, ?)", (set_id, pid, _now()))
                added += 1
            except aiosqlite.IntegrityError:
                pass  # уже в наборе
        await db.commit()
        return {"ok": True, "added": added}
    finally:
        await _release(db)


async def remove_word_from_set(user_id: int, set_id: int, pool_id: int):
    """Убрать слово из набора по pool_id. Прогресс SRS (user_words) не трогаем."""
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, set_id):
            return {"error": "Not found"}
        await db.execute("DELETE FROM dict_words WHERE dict_id = ? AND pool_id = ?", (set_id, pool_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def get_set_words(user_id: int, set_id: int):
    """Слова набора + статус изучения по каждому (для прогресса набора): None если набор чужой.
    status: mastered (выучено) | learning (в процессе) | new (ещё не трогали)."""
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, set_id):
            return None
        from .learning import required_cells   # ленивый импорт — избегаем цикла на загрузке модуля
        async with db.execute("""
            SELECT wp.id AS pool_id, wp.norwegian, wp.data, wp.level,
                   uw.mastered AS mastered, uw.correct AS correct, uw.incorrect AS incorrect,
                   uw.strength AS strength, uw.modes AS modes
            FROM dict_words dw
            JOIN word_pool wp ON wp.id = dw.pool_id
            LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
            WHERE dw.dict_id = ? ORDER BY dw.created_at DESC, dw.id DESC
        """, (user_id, set_id)) as cur:
            out = []
            for r in await cur.fetchall():
                data = json.loads(r["data"]) if r["data"] else {}
                attempts = (r["correct"] or 0) + (r["incorrect"] or 0)
                status = "mastered" if r["mastered"] == 1 else ("learning" if attempts > 0 else "new")
                try:
                    modes = json.loads(r["modes"]) if r["modes"] else {}
                except Exception:
                    modes = {}
                rcells = required_cells({"norwegian": r["norwegian"], "data": r["data"]})
                ramp = {"done": sum(1 for c in rcells if modes.get(c) == "1"), "total": len(rcells)}
                out.append({"pool_id": r["pool_id"], "norwegian": r["norwegian"],
                            "translate": data.get("translate", {}),
                            "part_of_speech": data.get("part_of_speech", ""), "level": r["level"],
                            "status": status, "ramp": ramp, "strength": r["strength"] or 0})
            return out
    finally:
        await _release(db)


async def reset_set_ramp(user_id: int, set_id: int):
    """Сбросить рампу ВЫУЧЕННЫХ слов набора — НЕ до интро-карточки, а до первого упражнения рампы
    (choice_int2no). Чистим клетки рампы, снимаем mastered/сертификацию, но оставляем
    attempts>0, чтобы интро-карточка (которая показывается при 0 попыток) не появлялась. due=сейчас."""
    from .learning import ALL_CELLS   # ленивый импорт (избегаем цикла на загрузке модуля)
    db = await _conn()
    try:
        if not await _owns_dict(db, user_id, set_id):
            return {"error": "Not found"}
        async with db.execute("""
            SELECT uw.id AS uid, uw.modes AS modes, uw.correct AS correct, uw.incorrect AS incorrect
            FROM dict_words dw
            JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = dw.pool_id
            WHERE dw.dict_id = ? AND uw.mastered = 1
        """, (user_id, set_id)) as cur:
            rows = [dict(r) for r in await cur.fetchall()]
    finally:
        await _release(db)

    # мутацию user_words — под пер-юзер локом; лок берём БЕЗ открытого соединения (порядок lock→conn
    # как в apply_result): держать permit пула, ожидая лок, = инверсия порядка и латентный дедлок.
    from .learning import _user_lock   # ленивый: тот же реестр локов, что у apply_result
    async with _user_lock(user_id):
        db = await _conn()
        try:
            for r in rows:
                try:
                    modes = json.loads(r["modes"]) if r["modes"] else {}
                except Exception:
                    modes = {}
                for c in ALL_CELLS:
                    modes[c] = ""
                corr = r["correct"] or 0
                if (corr + (r["incorrect"] or 0)) == 0:   # attempts==0 → поставим 1, чтобы не интро-карточка
                    corr = 1
                await db.execute(
                    "UPDATE user_words SET modes = ?, mastered = 0, certified = 0, was_certified = 0, "
                    "strength = 0, reps = 0, interval_days = 0, correct = ?, due_at = ? WHERE id = ?",
                    (json.dumps(modes, ensure_ascii=False), corr, _now(), r["uid"]))
            await db.commit()
        finally:
            await _release(db)
    return {"ok": True, "reset": len(rows)}


async def sets_for_words(user_id: int, pool_ids):
    """Карта pool_id → [set_id]: в каких личных наборах уже лежит каждое слово (для пикера)."""
    try:                              # нечисловой элемент → пустая карта, а не 500 глубже в SQL
        ids = [int(p) for p in (pool_ids or []) if p]
    except (TypeError, ValueError):
        return {}
    if not ids:
        return {}
    db = await _conn()
    try:
        marks = ",".join("?" for _ in ids)
        out = {}
        async with db.execute(f"""
            SELECT dw.pool_id, dw.dict_id FROM dict_words dw
            JOIN dictionaries d ON d.id = dw.dict_id
            WHERE d.user_id = ? AND COALESCE(d.hidden, 0) = 0 AND dw.pool_id IN ({marks})
        """, (user_id, *ids)) as cur:
            for r in await cur.fetchall():
                out.setdefault(r["pool_id"], []).append(r["dict_id"])
        return out
    finally:
        await _release(db)
