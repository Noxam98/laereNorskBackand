"""Аудит-фиксы G3: закрытие утечки approved=0 в vec/эмбеддинг-путь + жёсткая машина состояний
модерации/жалоб + ремап user_word_reports при мёрже дублей.

DB-тесты идут через fresh_db (свежая БД на тест, conftest; asyncio_mode=auto).
"""
import json

from db import (
    pool_missing_embedding, set_word_approval, resolve_report, report_word,
)
from db.pool_dedup import merge_pool_words
from db.core import _conn, _release, _now
from tests.conftest import seed_user, seed_word


# ── helpers ──────────────────────────────────────────────────────────────────
async def _insert_pool(no, *, ru="перевод", pos="noun", level="A1",
                       approved=1, embedding=None, reported=0, learn_excluded=0):
    """Вставить слово в пул напрямую (approved/embedding/reported под тест). Вернуть pool_id."""
    data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, level, approved, embedding, reported, "
            "learn_excluded, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (no, data, level, approved, embedding, reported, learn_excluded, _now()))
        pid = cur.lastrowid
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


async def _add_report(uid, pid):
    dbc = await _conn()
    try:
        await dbc.execute(
            "INSERT INTO user_word_reports (user_id, pool_id, created_at) VALUES (?,?,?)",
            (uid, pid, _now()))
        await dbc.commit()
    finally:
        await _release(dbc)


async def _reports_for(pid):
    """{user_id} — кто пожаловался на это слово (user_word_reports)."""
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT user_id FROM user_word_reports WHERE pool_id = ?", (pid,)) as cur:
            return {r["user_id"] for r in await cur.fetchall()}
    finally:
        await _release(dbc)


async def _pool_row(pid):
    dbc = await _conn()
    try:
        async with dbc.execute(
            "SELECT COALESCE(approved,1) approved, COALESCE(reported,0) reported, "
            "COALESCE(learn_excluded,0) learn_excluded FROM word_pool WHERE id = ?", (pid,)) as cur:
            return await cur.fetchone()
    finally:
        await _release(dbc)


async def _in_pool(pid):
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT 1 FROM word_pool WHERE id = ?", (pid,)) as cur:
            return (await cur.fetchone()) is not None
    finally:
        await _release(dbc)


# ── (a) эмбеддинг-путь исключает approved=0 ──────────────────────────────────
async def test_pool_missing_embedding_skips_unapproved(fresh_db):
    """Фоновый эмбеддер НЕ индексирует слово на модерации (approved=0) — иначе оно попало бы
    в vec_words и всплыло соседом-дистрактором (defense-in-depth к фильтру vec_nearest_rows)."""
    ok_pid = await _insert_pool("bil", approved=1, embedding=None)
    bad_pid = await _insert_pool("sykkel", approved=0, embedding=None)

    ids = [pid for pid, _no in await pool_missing_embedding(limit=50)]
    assert ok_pid in ids                                     # выверенное слово — на эмбеддинг
    assert bad_pid not in ids                                # approved=0 не индексируется


# ── (b) resolve_report: state-guard + очистка user_word_reports ──────────────
async def test_resolve_report_keep_on_non_reported_errors(fresh_db):
    _uid, did = await seed_user()
    pid, _ = await seed_word(did, "greitord")                # свежее слово: reported=0
    res = await resolve_report(pid, "keep")
    assert res.get("error") == "not_reported"               # «оставить» нечего — активной жалобы нет
    assert "ok" not in res


async def test_resolve_report_missing_pool_errors(fresh_db):
    res = await resolve_report(9_999_999, "keep")
    assert res.get("error") == "not_found"                   # несуществующий id — не молчим «успехом»


async def test_resolve_report_clears_user_word_reports(fresh_db):
    """Успешный вердикт чистит пер-юзер дедуп жалоб → цикл жалоб можно перезапустить."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "kanskjeord")
    r = await report_word(pid, uid)
    assert r["status"] == "queued"
    assert await _reports_for(pid) == {uid}                  # дедуп-строка появилась

    res = await resolve_report(pid, "keep")
    assert res.get("ok") is True
    assert await _reports_for(pid) == set()                  # очищена → та же когорта снова сможет репортить


# ── (c) set_word_approval: одобрять только слово на модерации ────────────────
async def test_set_word_approval_rejects_non_pending(fresh_db):
    pid = await _insert_pool("alt", approved=1)              # уже в общей базе (не на модерации)
    res = await set_word_approval(pid, 1)
    assert res.get("ok") is False
    assert res.get("error") == "not_pending"                 # повторное одобрение недопустимо


async def test_set_word_approval_approves_pending_and_clears_flags(fresh_db):
    """Одобрение слова на модерации переводит его в чистое состояние общей базы: снимаем
    возможную жалобу/исключение, чтобы approved=1 и learn_excluded=1 не сосуществовали."""
    pid = await _insert_pool("pending", approved=0, reported=1, learn_excluded=1)
    res = await set_word_approval(pid, 1)
    assert res.get("ok") is True
    row = await _pool_row(pid)
    assert row["approved"] == 1
    assert row["reported"] == 0                              # жалоба снята
    assert row["learn_excluded"] == 0                        # исключение снято


# ── (d) merge_pool_words ремапит user_word_reports на winner ─────────────────
async def test_merge_remaps_user_word_reports(fresh_db):
    uid, did = await seed_user()
    winner, _ = await seed_word(did, "hund", "собака")
    loser, _ = await seed_word(did, "hunden", "собака (опр.)")
    await _add_report(uid, loser)                            # жалоба «не учить» висит на loser'е

    assert await merge_pool_words(winner, loser) is True
    assert await _in_pool(loser) is False                    # loser удалён из пула
    assert await _reports_for(loser) == set()               # осиротевшей строки не осталось
    assert await _reports_for(winner) == {uid}              # жалоба переехала на winner
