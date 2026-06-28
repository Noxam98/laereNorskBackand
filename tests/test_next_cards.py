"""Живая сессия: добор новых карточек-знакомств по требованию (next_new_cards) — /learning/next-cards.
Когда юзер убирает карточку кнопкой (не «принял» тыком), подгружаем замену, пока не наберётся норма.
Проверяем форму элементов, исключение уже стоящих в очереди, и блок при закрытых воротах."""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning import next_new_cards, apply_result, PACK_FIRST
from tests.conftest import seed_user, seed_word


async def _add_base_word(no, ru="перевод", level="A1"):
    """Слово только в «Базу» (word_pool), ни в один словарь юзера → кандидат добора."""
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": "noun"})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)", (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _master(uid, pid):
    for mode, direction in (("choice", "no2int"), ("choice", "int2no"), ("build", "int2no"), ("input", "int2no")):
        await apply_result(uid, pid, True, mode=mode, direction=direction)


@pytest.mark.asyncio
async def test_next_cards_returns_intro_card_elements(fresh_db):
    uid, _ = await seed_user()
    for i in range(8):
        await _add_base_word(f"base{i}", f"пер{i}")
    res = await next_new_cards(uid, n=3)
    cards = res["cards"]
    assert len(cards) == 3
    for c in cards:                                  # форма ровно как у карточек build_session
        assert c["step"] == "card" and c["mode"] == "study" and c["direction"] is None
        assert c["pool_id"] and c["translate"]
        assert c["repeat"] is False
    assert len({c["pool_id"] for c in cards}) == 3   # три РАЗНЫХ новых слова


@pytest.mark.asyncio
async def test_next_cards_exclude_skips_queued(fresh_db):
    """exclude (уже стоящие в очереди pool_id) не повторяются в доборе."""
    uid, _ = await seed_user()
    for i in range(10):
        await _add_base_word(f"base{i}", f"пер{i}")
    first = await next_new_cards(uid, n=3)
    queued = [c["pool_id"] for c in first["cards"]]
    more = await next_new_cards(uid, n=3, exclude=queued)
    got = [c["pool_id"] for c in more["cards"]]
    assert got and set(got).isdisjoint(set(queued))


@pytest.mark.asyncio
async def test_next_cards_blocked_when_gate_closed(fresh_db):
    """Ворота закрыты (несданная пачка) → приток новых закрыт, добор пуст с blocked=True."""
    uid, did = await seed_user()
    for i in range(PACK_FIRST):
        pid, _ = await seed_word(did, f"pack{i}", f"пак{i}")
        await _master(uid, pid)
    for i in range(8):
        await _add_base_word(f"base{i}", f"пер{i}")
    res = await next_new_cards(uid, n=3)
    assert res.get("blocked") is True
    assert res["cards"] == []
