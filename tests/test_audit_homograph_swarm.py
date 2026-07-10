"""Омонимы на HTTP-границе (рой-аудит 10.07.2026).

Инвариант: запись пула определяется парой (norwegian, pos). Роут, который ПИШЕТ в запись или
строит по ней LLM-контекст, обязан принимать pool_id. Без него резолвер берёт ORDER BY id LIMIT 1
(старшего омонима) — и правка/описание/ответ уезжают в чужое значение слова.

Реальный кейс прода: 'mot' = preposition (id 578, «к/на») и noun (id 6752, «мужество»).
"""
import json

import pytest

from db.core import _conn, _release, _now
import db.pool as P


async def _seed_homographs():
    """Два омонима 'mot': предлог (МЛАДШИЙ id) и существительное (старший). Вернуть (prep, noun)."""
    db = await _conn()
    try:
        ids = []
        for pos, ru in (("preposition", "к"), ("noun", "мужество")):
            data = json.dumps({"translate": {"no": ["mot"], "ru": [ru]}, "part_of_speech": pos},
                              ensure_ascii=False)
            cur = await db.execute(
                "INSERT INTO word_pool (norwegian, data, pos, level, approved, created_at) "
                "VALUES (?,?,?,?,1,?)", ("mot", data, pos, "A1", _now()))
            ids.append(cur.lastrowid)
        await db.commit()
        return ids[0], ids[1]
    finally:
        await _release(db)


async def _desc_of(pid):
    db = await _conn()
    try:
        async with db.execute("SELECT description FROM word_pool WHERE id=?", (pid,)) as cur:
            r = await cur.fetchone()
            return json.loads(r["description"]) if r and r["description"] else None
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_redescribe_writes_into_requested_homograph(fresh_db, monkeypatch):
    """«Исправить описание» на карточке сущ. 'mot' не должно затирать описание предлога 'mot'."""
    import routers.pool as RP
    from models import RedescribeBody
    prep, noun = await _seed_homographs()
    assert prep < noun and await P.get_pool_id("mot") == prep   # дефолт-резолвер берёт предлог

    async def fake_ask(*a, **k):
        return {"description": {"ru": "отвага перед лицом опасности"}}
    monkeypatch.setattr(RP, "ask_json", fake_ask)

    await RP.pool_redescribe("mot", RedescribeBody(hint="это существительное"),
                             pool_id=noun, user={"id": 1, "username": "u"})
    assert (await _desc_of(noun))["ru"].startswith("отвага")
    assert await _desc_of(prep) is None      # предлог НЕ тронут


@pytest.mark.asyncio
async def test_ask_uses_requested_homograph_context(fresh_db, monkeypatch):
    """Вопрос про сущ. 'mot' должен уходить в LLM с контекстом существительного, а не предлога."""
    import routers.pool as RP
    from models import AskBody
    prep, noun = await _seed_homographs()
    seen = {}

    async def fake_ask(sys, ctx, schema, **k):
        seen["ctx"] = ctx
        return {"answer": "ok"}
    monkeypatch.setattr(RP, "ask_json", fake_ask)

    await RP.pool_ask("mot", AskBody(question="пример?", lang="ru"),
                      pool_id=noun, user={"id": 1, "username": "u"})
    assert "мужество" in seen["ctx"]                       # контекст от существительного
    seen.clear()
    await RP.pool_ask("mot", AskBody(question="пример?", lang="ru"),
                      pool_id=prep, user={"id": 1, "username": "u"})
    assert "мужество" not in seen["ctx"]                   # у предлога — свой контекст


@pytest.mark.asyncio
async def test_delete_pool_word_by_pool_id_keeps_other_homograph(fresh_db):
    """Админ удаляет ОДИН омоним из его карточки — второй должен остаться."""
    prep, noun = await _seed_homographs()
    await P.delete_pool_word("mot", pool_id=prep)
    assert await P.get_pool_by_id(prep) is None
    assert await P.get_pool_by_id(noun) is not None        # «мужество» цело


@pytest.mark.asyncio
async def test_delete_pool_word_without_pool_id_cleans_every_vector(fresh_db, monkeypatch):
    """Легаси-путь (без pool_id) сносит все омонимы — вектор надо чистить у КАЖДОГО,
    иначе второй остаётся призраком-дистрактором на несуществующий pool_id."""
    prep, noun = await _seed_homographs()
    killed = []

    async def fake_vec_delete(pid):
        killed.append(pid)
    monkeypatch.setattr(P, "vec_delete", fake_vec_delete)

    await P.delete_pool_word("mot")
    assert set(killed) == {prep, noun}     # раньше чистился только один произвольный id
    assert await P.get_pool_by_id(prep) is None and await P.get_pool_by_id(noun) is None


@pytest.mark.asyncio
async def test_delete_pool_word_cleans_compound_index(fresh_db):
    """word_pool_compounds заведена без FK/CASCADE → delete обязан чистить её явно (иначе сироты)."""
    from db.compound_index import set_pool_compounds
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, pos, approved, created_at) VALUES (?,?,?,1,?)",
            ("husbåt", json.dumps({"translate": {"no": ["husbåt"]}}), "noun", _now()))
        pid = cur.lastrowid
        await db.commit()
    finally:
        await _release(db)
    await set_pool_compounds([(pid, "husbåt", "hus", "båt")])
    await P.delete_pool_word("husbåt")
    db = await _conn()
    try:
        async with db.execute("SELECT 1 FROM word_pool_compounds WHERE pool_id=?", (pid,)) as cur:
            assert await cur.fetchone() is None
    finally:
        await _release(db)
