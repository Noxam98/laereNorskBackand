"""Центр уведомлений + передача набора другому пользователю.

Уведомление — ОБЩАЯ сущность на вырост: (type, payload JSON, state). Сейчас единственный тип —
`set_share`: «юзер X предлагает забрать набор Y». Принял → у получателя создаётся СВОЯ КОПИЯ
набора (он её редактирует как хочет), отклонил → предложение закрыто.

Ключевое решение: предложение несёт СНИМОК слов (pool_ids в payload), а не ссылку на набор
отправителя. Отправитель может потом переименовать/почистить/удалить свой набор — предложение
от этого не протухнет и не утащит лишнего. Снимок капается (MAX_SHARE_WORDS).

Приватность общего пула: слова кладёт add_words_to_set, а он пропускает через _pool_visible —
чужое неодобренное слово (approved=0) в копию не попадёт, даже если отправитель его видел.
"""
import json
from .core import _conn, _release, _now

MAX_SHARE_WORDS = 500      # потолок снимка набора (и защита payload от раздувания)
PENDING_LIMIT = 20         # сколько НЕрешённых предложений юзер может держать на одного получателя
LIST_LIMIT = 50            # сколько уведомлений отдаём в панель


def _payload(row):
    try:
        return json.loads(row["payload"]) if row["payload"] else {}
    except Exception:
        return {}


def _shape(row):
    """Уведомление наружу: payload разворачиваем, но pool_ids не отдаём (клиенту они не нужны,
    а список чужих слов — лишние данные в ответе)."""
    p = _payload(row)
    p.pop("pool_ids", None)
    return {"id": row["id"], "type": row["type"], "state": row["state"],
            "read": bool(row["read_at"]), "created_at": row["created_at"], **p}


async def create_notification(user_id: int, type_: str, payload: dict):
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO notifications (user_id, type, payload, state, created_at) VALUES (?,?,?,'new',?)",
            (user_id, type_, json.dumps(payload or {}, ensure_ascii=False), _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def list_notifications(user_id: int, limit: int = LIST_LIMIT):
    """Лента уведомлений (новые сверху) + счётчик непрочитанных для бейджа."""
    limit = max(1, min(LIST_LIMIT, int(limit or LIST_LIMIT)))
    db = await _conn()
    try:
        async with db.execute(
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)) as cur:
            items = [_shape(r) for r in await cur.fetchall()]
        async with db.execute(
            "SELECT COUNT(*) c FROM notifications WHERE user_id = ? AND read_at IS NULL", (user_id,)) as cur:
            unread = (await cur.fetchone())["c"]
        return {"items": items, "unread": unread}
    finally:
        await _release(db)


async def mark_read(user_id: int):
    """Панель открыли — гасим бейдж. Решение по предложению (accept/decline) это НЕ трогает:
    непрочитанность и незакрытость — разные вещи."""
    db = await _conn()
    try:
        await db.execute("UPDATE notifications SET read_at = ? WHERE user_id = ? AND read_at IS NULL",
                         (_now(), user_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


async def _pending_count(db, from_id: int, to_id: int):
    async with db.execute(
        "SELECT COUNT(*) c FROM notifications WHERE user_id = ? AND type = 'set_share' AND state = 'new' "
        "AND json_extract(payload, '$.from_id') = ?", (to_id, from_id)) as cur:
        return (await cur.fetchone())["c"]


async def share_set(user_id: int, set_id: int, target_id: int):
    """Предложить набор другому пользователю. Проверяем: набор МОЙ и непустой, получатель
    существует и это не я сам, и я не заваливаю его предложениями (PENDING_LIMIT)."""
    if target_id == user_id:
        return {"error": "self"}
    db = await _conn()
    try:
        async with db.execute("SELECT name FROM dictionaries WHERE id = ? AND user_id = ? AND COALESCE(hidden,0) = 0",
                              (set_id, user_id)) as cur:
            src = await cur.fetchone()
        if not src:
            return {"error": "Not found"}
        async with db.execute("SELECT id, display_name, username FROM users WHERE id = ?", (target_id,)) as cur:
            target = await cur.fetchone()
        if not target:
            return {"error": "no user"}
        async with db.execute("SELECT display_name, username FROM users WHERE id = ?", (user_id,)) as cur:
            me = await cur.fetchone()
        async with db.execute(
            "SELECT pool_id FROM dict_words WHERE dict_id = ? ORDER BY created_at, id LIMIT ?",
            (set_id, MAX_SHARE_WORDS)) as cur:
            pool_ids = [r["pool_id"] for r in await cur.fetchall()]
        if not pool_ids:
            return {"error": "empty"}
        if await _pending_count(db, user_id, target_id) >= PENDING_LIMIT:
            return {"error": "too many"}
    finally:
        await _release(db)
    payload = {
        "set_name": src["name"],
        "count": len(pool_ids),
        "from_id": user_id,
        "from_name": (me["display_name"] if me and me["display_name"] else (me["username"] if me else "")),
        "pool_ids": pool_ids,
    }
    nid = await create_notification(target_id, "set_share", payload)
    return {"ok": True, "id": nid, "count": len(pool_ids)}


def share_push_text(sender_row, count: int):
    """Текст адресного пуша о предложении набора. Язык получателя на бэке не знаем (настройка
    интерфейса живёт в gamePrefs и меняется на лету) — шлём по-русски, как и прочие наши пуши."""
    who = (sender_row.get("display_name") or sender_row.get("username") or "").strip() if sender_row else ""
    return ("Вам прислали набор слов", f"{who or 'Пользователь'} предлагает набор ({count} слов)")


async def _unique_name(db, user_id: int, name: str):
    """Имя копии: как у отправителя, а при конфликте — «Имя (2)», «Имя (3)»… (у dictionaries
    уникальность на (user_id, name), и без этого вторая копия просто не создалась бы)."""
    base = (name or "Набор").strip()[:60] or "Набор"
    async with db.execute("SELECT name FROM dictionaries WHERE user_id = ?", (user_id,)) as cur:
        taken = {r["name"] for r in await cur.fetchall()}
    if base not in taken:
        return base
    for n in range(2, 100):
        cand = f"{base} ({n})"
        if cand not in taken:
            return cand
    return f"{base} ({_now()[:19]})"


async def decide_share(user_id: int, nid: int, accept: bool):
    """Принять/отклонить предложение набора. Принятие создаёт СВОЮ копию (studying=0 — чужой
    набор не должен сам лезть в ежедневные сессии) и помечает уведомление accepted."""
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM notifications WHERE id = ? AND user_id = ?", (nid, user_id)) as cur:
            row = await cur.fetchone()
        if not row or row["type"] != "set_share":
            return {"error": "Not found"}
        if row["state"] != "new":
            return {"error": "decided", "state": row["state"]}
        payload = _payload(row)
    finally:
        await _release(db)

    if not accept:
        db = await _conn()
        try:
            await db.execute("UPDATE notifications SET state = 'declined' WHERE id = ? AND user_id = ?", (nid, user_id))
            await db.commit()
        finally:
            await _release(db)
        return {"ok": True, "state": "declined"}

    from .dictionaries import create_dictionary
    from .sets_data import add_words_to_set
    db = await _conn()
    try:
        name = await _unique_name(db, user_id, payload.get("set_name"))
    finally:
        await _release(db)
    created = await create_dictionary(user_id, name)
    if created.get("error"):
        return {"error": created["error"]}
    set_id = created["id"]
    # слова кладём штатным путём — он же чистит дубли и режет чужие неодобренные (_pool_visible)
    res = await add_words_to_set(user_id, set_id, payload.get("pool_ids") or [])
    db = await _conn()
    try:
        await db.execute("UPDATE notifications SET state = 'accepted' WHERE id = ? AND user_id = ?", (nid, user_id))
        await db.commit()
    finally:
        await _release(db)
    return {"ok": True, "state": "accepted", "set_id": set_id, "name": name,
            "added": res.get("added", 0) if isinstance(res, dict) else 0}
