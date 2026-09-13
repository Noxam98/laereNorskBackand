"""Передача набора другому пользователю + центр уведомлений.

Модель: предложение несёт СНИМОК слов (pool_ids в payload), принятие создаёт СВОЮ копию набора
(получатель её редактирует как хочет), отклонение закрывает предложение. Отдельно проверяем,
что чужое неодобренное слово (approved=0) в копию не протекает — копию наполняет
add_words_to_set, а он гейтит видимость.
"""
import json
from db.core import _conn, _release, _now
from db.notifications import share_set, list_notifications, mark_read, decide_share
from db.users import search_users
from db.sets_data import list_user_sets, get_set_words
from tests.conftest import seed_user, seed_word


async def _mk_set(user_id, name):
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO dictionaries (user_id,name,created_at,studying) VALUES (?,?,?,0)",
                               (user_id, name, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _add(set_id, pool_id):
    db = await _conn()
    try:
        await db.execute("INSERT INTO dict_words (dict_id,pool_id,created_at) VALUES (?,?,?)",
                         (set_id, pool_id, _now()))
        await db.commit()
    finally:
        await _release(db)


async def _named(uid, name):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET display_name = ? WHERE id = ?", (name, uid))
        await db.commit()
    finally:
        await _release(db)


async def _pair():
    """Отправитель с набором из двух слов + получатель."""
    a_uid, a_did = await seed_user("sender")
    b_uid, _b_did = await seed_user("receiver")
    w1, _ = await seed_word(a_did, "bil", "машина")
    w2, _ = await seed_word(a_did, "hus", "дом")
    sid = await _mk_set(a_uid, "Теория")
    await _add(sid, w1)
    await _add(sid, w2)
    return a_uid, b_uid, sid, (w1, w2)


async def test_share_creates_notification(fresh_db):
    a, b, sid, _w = await _pair()
    await _named(a, "Максим")
    res = await share_set(a, sid, b)
    assert res["ok"] and res["count"] == 2
    box = await list_notifications(b)
    assert box["unread"] == 1
    n = box["items"][0]
    assert n["type"] == "set_share" and n["state"] == "new"
    assert n["set_name"] == "Теория" and n["count"] == 2 and n["from_name"] == "Максим"
    assert "pool_ids" not in n            # список чужих слов наружу не отдаём
    assert (await list_notifications(a))["items"] == []   # отправителю уведомления нет


async def test_accept_makes_own_editable_copy(fresh_db):
    a, b, sid, (w1, w2) = await _pair()
    await share_set(a, sid, b)
    nid = (await list_notifications(b))["items"][0]["id"]
    res = await decide_share(b, nid, True)
    assert res["state"] == "accepted" and res["added"] == 2
    sets = await list_user_sets(b)
    copy = [s for s in sets if s["name"] == "Теория"]
    assert len(copy) == 1 and copy[0]["count"] == 2
    assert copy[0]["studying"] is False        # чужой набор сам в ежедневную учёбу не лезет
    assert copy[0]["id"] != sid                # это СВОЙ набор, а не ссылка на чужой
    words = await get_set_words(b, copy[0]["id"])
    assert {w["pool_id"] for w in words} == {w1, w2}
    assert (await list_notifications(b))["items"][0]["state"] == "accepted"


async def test_copy_is_independent_from_sender(fresh_db):
    """Отправитель потом чистит/удаляет свой набор — у получателя копия целая."""
    a, b, sid, _w = await _pair()
    await share_set(a, sid, b)
    nid = (await list_notifications(b))["items"][0]["id"]
    await decide_share(b, nid, True)
    db = await _conn()
    try:
        await db.execute("DELETE FROM dict_words WHERE dict_id = ?", (sid,))
        await db.execute("DELETE FROM dictionaries WHERE id = ?", (sid,))
        await db.commit()
    finally:
        await _release(db)
    copy = [s for s in await list_user_sets(b) if s["name"] == "Теория"][0]
    assert copy["count"] == 2


async def test_decline_closes_offer(fresh_db):
    a, b, sid, _w = await _pair()
    await share_set(a, sid, b)
    nid = (await list_notifications(b))["items"][0]["id"]
    assert (await decide_share(b, nid, False))["state"] == "declined"
    assert [s for s in await list_user_sets(b) if s["name"] == "Теория"] == []
    # решение принимается ОДИН раз — повторное «принять» уже не создаст набор
    again = await decide_share(b, nid, True)
    assert again["error"] == "decided"
    assert [s for s in await list_user_sets(b) if s["name"] == "Теория"] == []


async def test_name_collision_gets_suffix(fresh_db):
    a, b, sid, _w = await _pair()
    await _mk_set(b, "Теория")                 # у получателя уже есть набор с таким именем
    await share_set(a, sid, b)
    nid = (await list_notifications(b))["items"][0]["id"]
    res = await decide_share(b, nid, True)
    assert res["name"] == "Теория (2)"


async def test_foreign_unapproved_word_does_not_leak(fresh_db):
    """Приватное неодобренное слово отправителя в чужую копию не попадает (гейт видимости)."""
    a, b, sid, _w = await _pair()
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at,approved,created_by) VALUES (?,?,?,?,0,?)",
            ("hemmelig", json.dumps({"translate": {"ru": ["секрет"]}}), "B1", _now(), a))
        secret = cur.lastrowid
        await db.commit()
    finally:
        await _release(db)
    await _add(sid, secret)
    await share_set(a, sid, b)
    nid = (await list_notifications(b))["items"][0]["id"]
    await decide_share(b, nid, True)
    copy = [s for s in await list_user_sets(b) if s["name"] == "Теория"][0]
    words = await get_set_words(b, copy["id"])
    assert secret not in {w["pool_id"] for w in words}
    assert len(words) == 2


async def test_cannot_share_alien_or_empty_set(fresh_db):
    a, b, _sid, _w = await _pair()
    empty = await _mk_set(a, "Пусто")
    assert (await share_set(a, empty, b))["error"] == "empty"
    b_set = await _mk_set(b, "Чужой")
    assert (await share_set(a, b_set, b))["error"] == "Not found"   # не мой набор
    assert (await share_set(a, _sid, a))["error"] == "self"


async def test_read_clears_badge_but_not_decision(fresh_db):
    a, b, sid, _w = await _pair()
    await share_set(a, sid, b)
    assert (await list_notifications(b))["unread"] == 1
    await mark_read(b)
    box = await list_notifications(b)
    assert box["unread"] == 0
    assert box["items"][0]["state"] == "new"    # прочитано ≠ решено: кнопки остаются
    assert box["items"][0]["read"] is True


async def test_user_search(fresh_db):
    a, _b, _sid, _w = await _pair()
    await _named(a, "Максим Мельников")
    found = await search_users("макс", exclude_id=None)
    assert [u["name"] for u in found] == ["Максим Мельников"]
    assert set(found[0].keys()) == {"id", "name"}        # ни почты, ни логина, ни прогресса
    assert await search_users("м", exclude_id=None) == []      # слишком короткий запрос
    assert await search_users("sender", exclude_id=a) == []    # себя не показываем
    assert [u["id"] for u in await search_users("sender")] == [a]   # по логину точным вводом — находится
