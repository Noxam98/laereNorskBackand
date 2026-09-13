"""Фильтр Базы «(не) в моём наборе» — срез для экрана добора слов в набор.

Ключевое: владение набором зашито в сам SQL (_set_cond), поэтому чужой set_id не может ни
показать чужую подборку, ни отфильтровать выдачу. Плюс флаг inSet на слове страницы —
по нему экран красит карточку «уже добавлено».
"""
from db.core import _conn, _release, _now
from db.pool import get_pool_list
from tests.conftest import seed_user, seed_word


async def _mk_set(user_id, name="Контрольная"):
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO dictionaries (user_id,name,created_at) VALUES (?,?,?)",
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


async def _seed(uid, did):
    a, _ = await seed_word(did, "bil", "машина")
    b, _ = await seed_word(did, "hus", "дом")
    c, _ = await seed_word(did, "katt", "кот")
    return a, b, c


async def test_in_set_and_out_of_set(fresh_db):
    uid, did = await seed_user()
    a, b, c = await _seed(uid, did)
    sid = await _mk_set(uid)
    await _add(sid, a)
    await _add(sid, b)

    inside = await get_pool_list(user_id=uid, set_id=sid, in_set=True)
    assert {w["pool_id"] for w in inside["words"]} == {a, b}
    assert inside["total"] == 2

    outside = await get_pool_list(user_id=uid, set_id=sid, in_set=False)
    assert {w["pool_id"] for w in outside["words"]} == {c}
    assert outside["total"] == 1


async def test_in_set_flag_without_filter(fresh_db):
    """Фильтр выключен (in_set=None) — список полный, но каждое слово знает, оно в наборе или нет."""
    uid, did = await seed_user()
    a, _b, _c = await _seed(uid, did)
    sid = await _mk_set(uid)
    await _add(sid, a)
    res = await get_pool_list(user_id=uid, set_id=sid)
    assert res["total"] == 3
    flags = {w["pool_id"]: w["inSet"] for w in res["words"]}
    assert flags[a] is True and all(v is False for k, v in flags.items() if k != a)


async def test_no_set_id_no_flag(fresh_db):
    """Обычная База (без set_id) отдаёт ту же форму, что и раньше — лишнего ключа нет."""
    uid, did = await seed_user()
    await _seed(uid, did)
    res = await get_pool_list(user_id=uid)
    assert res["words"] and all("inSet" not in w for w in res["words"])


async def test_foreign_set_cannot_be_peeked(fresh_db):
    """Чужой набор: фильтр «в наборе» не показывает НИЧЕГО (а не чужие слова),
    «не в наборе» — весь пул. Подглядеть чужую подборку по id нельзя."""
    uid, did = await seed_user()
    other, other_did = await seed_user("other")
    a, b, c = await _seed(uid, did)
    foreign = await _mk_set(other, "чужой")
    await _add(foreign, a)

    inside = await get_pool_list(user_id=uid, set_id=foreign, in_set=True)
    assert inside["words"] == [] and inside["total"] == 0

    outside = await get_pool_list(user_id=uid, set_id=foreign, in_set=False)
    assert {w["pool_id"] for w in outside["words"]} == {a, b, c}
    assert all(w["inSet"] is False for w in outside["words"])
    assert other_did  # словарь второго юзера заведён (фикстура не пустая)


async def test_set_filter_combines_with_other_filters(fresh_db):
    """Срез по набору не отменяет прочие фильтры Базы (уровень/поиск) — они складываются."""
    uid, did = await seed_user()
    a, _ = await seed_word(did, "bil", "машина", level="A1")
    b, _ = await seed_word(did, "hus", "дом", level="B2")
    sid = await _mk_set(uid)
    await _add(sid, a)
    res = await get_pool_list(user_id=uid, set_id=sid, in_set=False, level="B2")
    assert [w["pool_id"] for w in res["words"]] == [b]
    res2 = await get_pool_list(user_id=uid, set_id=sid, in_set=False, q="bil")
    assert res2["words"] == []          # bil в наборе → срез «не в наборе» его не отдаёт
