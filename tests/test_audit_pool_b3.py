"""Аудит-фиксы B3: stale-эмбеддинг после правки + вычитка дистракторов по approved.

DB-тесты идут через fresh_db (свежая БД на тест, conftest).
(Разделение фаз слов/форм — осознанное, покрыто test_learning_forms.py::
test_form_reviews_live_in_forms_phase; здесь не дублируем.)
"""
import json

from db import (
    get_pool_by_id, update_pool_word, load_pool_embeddings, pool_by_freq,
)
from db.core import _conn, _release, _now


# ── helpers ──────────────────────────────────────────────────────────────────
async def _insert_pool(no, *, ru="перевод", pos="noun", level="A1", freq=5.0,
                       approved=1, embedding=b"\x01\x02"):
    """Вставить слово в пул напрямую (approved/embedding под тест). Вернуть pool_id."""
    data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, level, freq, approved, embedding, created_at) "
            "VALUES (?,?,?,?,?,?,?)", (no, data, level, freq, approved, embedding, _now()))
        pid = cur.lastrowid
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


# ── #1 update_pool_word зануляет ХРАНИМЫЙ эмбеддинг ───────────────────────────
async def test_update_pool_word_nulls_persisted_embedding(fresh_db):
    pid = await _insert_pool("hus", embedding=b"\xaa\xbb\xcc")
    before = await get_pool_by_id(pid)
    assert before["embedding"] == b"\xaa\xbb\xcc"           # эмбеддинг был

    res = await update_pool_word("hus", {"ru": ["дом"]})
    assert res.get("ok") is True

    after = await get_pool_by_id(pid)
    assert after["embedding"] is None                        # стёрт → не всплывёт на рестарте


# ── #3 дистракторные запросы исключают approved=0 ────────────────────────────
async def test_load_pool_embeddings_excludes_unapproved(fresh_db):
    ok_pid = await _insert_pool("bil", approved=1, embedding=b"\x11\x22")
    bad_pid = await _insert_pool("sykkel", approved=0, embedding=b"\x33\x44")

    ids, embs = await load_pool_embeddings()
    assert ok_pid in ids                                     # вычитанное слово — кандидат
    assert bad_pid not in ids                                # approved=0 не идёт в дистракторы
    assert len(ids) == len(embs)


async def test_pool_by_freq_excludes_unapproved(fresh_db):
    ok_pid = await _insert_pool("katt", approved=1)
    bad_pid = await _insert_pool("hund", approved=0)

    rows = await pool_by_freq(limit=50, level="A1")
    got = {r["pool_id"] for r in rows}
    assert ok_pid in got
    assert bad_pid not in got                                # approved=0 не всплывает в частотном отборе
