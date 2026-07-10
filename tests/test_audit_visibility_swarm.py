"""Видимость модерации у ВСЕХ читателей/писателей пула (рой-аудит 10.07.2026).

Инвариант: чужое неодобренное слово (approved=0, created_by=A) не должно попадать к юзеру B
НИ через выдачу, НИ через кросс-юзерный контент (онлайн-игра, плейсмент, дистракторы, подбор
новых слов/фраз, разблокировка композитов), НИ через приёмник pool_id («добавить в словарь/набор»).

Правило: для кросс-юзерного контента (единственного user_id нет) — ЖЁСТКО approved=1.
Для персональных выдач — видимость (approved=1 OR created_by=user).
Внутренние фоновые вызовы фильтр не применяют осознанно (граница безопасности = HTTP-роут).
"""
import json

import pytest

from db.core import _conn, _release, _now
from tests.conftest import seed_user

A, B = 901, 902   # авторы: A владеет pending-словом, B — «другой юзер»


async def _insert(no, *, ru="перевод", pos="noun", level="A1", approved=1, created_by=None,
                  freq=5.0, data_extra=None, embedding=b"\x01\x02"):
    d = {"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos}
    if data_extra:
        d.update(data_extra)
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, freq, approved, created_by, embedding, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (no, json.dumps(d, ensure_ascii=False), pos, level, freq, approved, created_by, embedding, _now()))
        pid = cur.lastrowid
        await db.commit()
        return pid
    finally:
        await _release(db)


async def _topic(pid, topic="food"):
    db = await _conn()
    try:
        await db.execute("INSERT INTO word_topics (pool_id, topic) VALUES (?,?)", (pid, topic))
        await db.commit()
    finally:
        await _release(db)


# ── КОРЕНЬ: приёмник pool_id (IDOR перебором id) ──────────────────────────────
@pytest.mark.asyncio
async def test_add_word_to_dict_rejects_other_users_pending(fresh_db):
    """POST /learning/add {pool_id: <чужой approved=0>} — id последовательны, перебор тривиален.
    Раньше приёмник проверял только владение словарём, а не видимость слова: чужое pending
    ложилось в скрытый словарь B, и его содержимое читалось внутренними путями сессии/карточки."""
    from db.dictionaries import add_word_to_dict, get_or_create_hidden_dict
    uid_b, _ = await seed_user("bob")
    mine = await _insert("mitt", approved=0, created_by=uid_b)     # СВОЁ неодобренное — можно
    alien = await _insert("hemmelig", approved=0, created_by=A)    # чужое — нельзя
    shared = await _insert("hus", approved=1)                      # общая база — можно

    hid = await get_or_create_hidden_dict(uid_b)
    assert (await add_word_to_dict(uid_b, hid, shared)).get("error") is None
    assert (await add_word_to_dict(uid_b, hid, mine)).get("error") is None
    assert (await add_word_to_dict(uid_b, hid, alien)).get("error") == "Not found"


@pytest.mark.asyncio
async def test_add_words_to_set_skips_other_users_pending(fresh_db):
    """/sets/{id}/words вставлял сырые клиентские pool_ids без проверки видимости."""
    from db.sets_data import add_words_to_set
    from db.dictionaries import create_dictionary
    uid_b, _ = await seed_user("bob2")
    alien = await _insert("hemmelig", approved=0, created_by=A)
    shared = await _insert("hus", approved=1)
    sid = (await create_dictionary(uid_b, "мой набор"))["id"]

    res = await add_words_to_set(uid_b, sid, [shared, alien])
    assert res["added"] == 1     # чужое pending пропущено

    db = await _conn()
    try:
        async with db.execute("SELECT pool_id FROM dict_words WHERE dict_id=?", (sid,)) as cur:
            got = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert got == {shared}


# ── кросс-юзерный контент: жёстко approved=1 ─────────────────────────────────
@pytest.mark.asyncio
async def test_duel_words_exclude_pending(fresh_db):
    """Онлайн-викторина (source='pool' по умолчанию) и входной тест: вопросы/дистракторы видят
    ВСЕ игроки, поэтому личные расширения (approved=0) недопустимы в принципе."""
    from db.pool import get_pool_duel_words
    await _insert("hus", approved=1, level="A1")
    await _insert("hemmelig", approved=0, created_by=A, level="A1")
    words = {w["norwegian"] for w in await get_pool_duel_words(50, "A1", None)}
    assert "hus" in words and "hemmelig" not in words


@pytest.mark.asyncio
async def test_pool_by_freq_topics_excludes_pending(fresh_db):
    """Подбор новых слов по темам идёт ДРУГИМ юзерам — чужое pending не должно всплывать в Учёбе."""
    from db.pool_freq import pool_by_freq_topics
    ok = await _insert("hus", approved=1, level="A1", freq=6.0)
    bad = await _insert("hemmelig", approved=0, created_by=A, level="A1", freq=6.5)
    await _topic(ok); await _topic(bad)
    ids = {w["pool_id"] for w in await pool_by_freq_topics(20, "A1", ["food"])}
    assert ok in ids and bad not in ids


@pytest.mark.asyncio
async def test_suggest_phrases_excludes_pending(fresh_db):
    """Фразы авто-подмешиваются в Учёбу другим юзерам (третий сиблинг pool_by_freq*)."""
    from db.learning import suggest_phrases
    uid, _ = await seed_user("carl")
    game = {"game": {"distractors": ["toget", "bilen"]}}
    for d in ("toget", "bilen"):
        await _insert(d, level="A1")
    ok = await _insert("ta bussen", pos="phrase", level="A1", approved=1, data_extra=game)
    bad = await _insert("gi beskjed", pos="phrase", level="A1", approved=0, created_by=A, data_extra=game)

    await suggest_phrases(uid, count=10, level="A2")
    db = await _conn()
    try:
        async with db.execute(
                "SELECT dw.pool_id FROM dict_words dw JOIN dictionaries d ON d.id=dw.dict_id "
                "WHERE d.user_id=?", (uid,)) as cur:
            owned = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert ok in owned and bad not in owned


@pytest.mark.asyncio
async def test_get_pool_words_by_ids_excludes_pending(fresh_db):
    """Дистракторы из резидентного embcache: вектор approved=0 попадает в кеш синхронно при
    persist_pool и живёт до рестарта → фильтр обязан стоять на чтении слов по id."""
    from db.pool import get_pool_words_by_ids
    ok = await _insert("hus", approved=1)
    bad = await _insert("hemmelig", approved=0, created_by=A)
    got = await get_pool_words_by_ids([ok, bad])
    assert ok in got and bad not in got


@pytest.mark.asyncio
async def test_pool_list_fuzzy_fallback_hides_other_users_pending(fresh_db):
    """Опечатка → подстрока не нашла → fuzzy-слой. Он строит индекс по ВСЕМУ пулу, поэтому SELECT
    по его id обязан фильтровать видимость (та же дыра, что чинили в search_pool)."""
    from db.pool import get_pool_list
    await _insert("kjøkken", ru="кухня", approved=0, created_by=A)
    res = await get_pool_list(60, 0, "kjoken", [], None, "alpha", "asc", None, None,
                              user_id=B, lang="ru")
    assert [w["word"] for w in res.get("words", [])] == []      # чужому — не видно
    res_own = await get_pool_list(60, 0, "kjoken", [], None, "alpha", "asc", None, None,
                                  user_id=A, lang="ru")
    assert "kjøkken" in [w["word"] for w in res_own.get("words", [])]   # автору — видно


@pytest.mark.asyncio
async def test_compound_index_excludes_pending(fresh_db):
    """index/learnable глобальны (их видят все) → чужой pending-композит не должен
    разблокироваться в Учёбе у других через suggest_compounds."""
    from db.compound_index import set_pool_compounds, load_index, invalidate
    await _insert("hus"); await _insert("båt")
    ok = await _insert("husbåt", approved=1)
    bad = await _insert("hemmelighus", approved=0, created_by=A)
    await set_pool_compounds([(ok, "husbåt", "hus", "båt"), (bad, "hemmelighus", "hemmelig", "hus")])
    invalidate()
    index, _feat, _learnable = await load_index()
    ids = {e["pool_id"] for e in index}
    assert ok in ids and bad not in ids
