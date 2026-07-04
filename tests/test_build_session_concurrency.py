"""ЭТАП 9: пер-юзер лок build_session — два таба не досыпают дважды.

Гонка: у юзера пул новых пуст (new_avail==0), два одновременных build_session
видят пустоту и оба зовут suggest_words → дубль-вставки. Пер-юзер лок + перечитка
под локом: второй ждущий видит уже досыпанное первым и НЕ добирает снова.
"""
import asyncio

from db import learning
from db.core import _conn, _release
from db.dictionaries import get_or_create_hidden_dict, add_word_to_dict
from tests.conftest import seed_user, seed_word


def test_user_lock_registry_is_per_user():
    a1 = learning._user_lock(101)
    a2 = learning._user_lock(101)
    b = learning._user_lock(202)
    assert a1 is a2 and a1 is not b   # тот же юзер → тот же лок; разные → разные


async def test_concurrent_build_supplies_once(fresh_db, monkeypatch):
    uid, did = await seed_user("conc")
    # запасное слово в ПУЛЕ, но НЕ в словаре юзера (его добросит фейковый suggest_words)
    spare_pid, _ = await seed_word(did, "reserve", ru="запас")
    dbc = await _conn()
    try:
        await dbc.execute("DELETE FROM dict_words WHERE pool_id = ?", (spare_pid,))
        await dbc.commit()
    finally:
        await _release(dbc)

    calls = {"n": 0}

    async def fake_suggest(user_id, count=10, level=None, allow_func=True):
        calls["n"] += 1
        hid = await get_or_create_hidden_dict(user_id)
        await add_word_to_dict(user_id, hid, spare_pid)   # реально добавляем → reload увидит new_avail>0
        return {"added": 1, "words": [{"pool_id": spare_pid}], "level": "A1"}

    monkeypatch.setattr(learning, "suggest_words", fake_suggest)
    monkeypatch.setattr(learning, "estimate_level", lambda uid: _coro("A1"))
    # фразы не мешают: держим буфер «полным»
    monkeypatch.setattr(learning, "_phrase_supply", lambda db, uid: _coro(999))

    await asyncio.gather(learning.build_session(uid, size=10),
                         learning.build_session(uid, size=10))
    assert calls["n"] == 1   # второй под локом перечитал состояние и НЕ добрал повторно


async def _coro(v):
    return v
