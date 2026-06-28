"""Жалобы «не учить»: очередь админа → exclude / keep + авто-гашение следующих 5 жалоб."""
import pytest

from db.core import _conn, _release, _now
from db import report_word, reported_words, reported_count, resolve_report, pool_by_freq


async def _add_user_word(uid, pid):
    dbc = await _conn()
    try:
        await dbc.execute("INSERT INTO user_words (user_id,pool_id,created_at) VALUES (?,?,?)", (uid, pid, _now()))
        await dbc.commit()
    finally:
        await _release(dbc)


async def _has_user_word(uid, pid):
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT COUNT(*) c FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            return (await cur.fetchone())["c"] > 0
    finally:
        await _release(dbc)


@pytest.mark.asyncio
async def test_report_lifecycle_queue_keep_dismiss_exclude(fresh_db):
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "pokemon")
    await _add_user_word(uid, pid)

    # 1) первая жалоба → в очередь админа, и слово убрано из Учёбы пользователя
    r = await report_word(pid, uid)
    assert r["status"] == "queued"
    assert not await _has_user_word(uid, pid)
    assert await reported_count() == 1
    assert [w["pool_id"] for w in await reported_words()] == [pid]

    # 2) админ «оставить» → жалоба снята, выбор запомнен на следующие 5 жалоб
    await resolve_report(pid, "keep")
    assert await reported_count() == 0

    # 3) следующие 5 жалоб гасятся автоматически (не идут в очередь)
    for _ in range(5):
        assert (await report_word(pid, uid))["status"] == "dismissed"
    assert await reported_count() == 0

    # 4) 6-я жалоба (память исчерпана) → снова в очередь админа
    assert (await report_word(pid, uid))["status"] == "queued"
    assert await reported_count() == 1

    # 5) админ «убрать из учёбы» → слово исчезает из подбора новых
    assert "pokemon" in [w["norwegian"] for w in await pool_by_freq(100)]
    await resolve_report(pid, "exclude")
    assert await reported_count() == 0
    assert "pokemon" not in [w["norwegian"] for w in await pool_by_freq(100)]


@pytest.mark.asyncio
async def test_report_on_already_excluded_is_silent(fresh_db):
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "junk")
    await resolve_report(pid, "exclude")
    r = await report_word(pid, uid)
    assert r["status"] == "excluded"
    assert await reported_count() == 0


@pytest.mark.asyncio
async def test_report_puts_word_in_personal_skip_and_unlinks(fresh_db):
    """«Отправить на модерацию» → персональная свалка + слово убрано из словарей юзера."""
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "junkx")
    await report_word(pid, uid)
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT COUNT(*) c FROM user_word_skips WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            assert (await cur.fetchone())["c"] == 1   # в свалке
        async with dbc.execute("SELECT COUNT(*) c FROM dict_words WHERE pool_id=? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id=?)", (pid, uid)) as cur:
            assert (await cur.fetchone())["c"] == 0   # отвязано от словарей
        async with dbc.execute("SELECT COUNT(*) c FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            assert (await cur.fetchone())["c"] == 0   # прогресс удалён (не «выучено»)
    finally:
        await _release(dbc)


@pytest.mark.asyncio
async def test_known_marks_mastered_not_archived(fresh_db):
    """«Уже знаю» → в Выучено (mastered), НЕ в архив."""
    from db import learning_set_status, learning_stats
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "kjentord")
    await learning_set_status(uid, pid, "known")
    st = await learning_stats(uid)
    assert st["byStatus"].get("mastered", 0) >= 1
    assert st["byStatus"].get("archived", 0) == 0


@pytest.mark.asyncio
async def test_skip_word_removes_personally_without_moderation(fresh_db):
    """«Не актуально» (skip): персональная свалка + отвязка от словарей, но БЕЗ модерации —
    жалоба не заводится, слово остаётся в общей базе (в отличие от report)."""
    from db import skip_word
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "neaktuelt")
    await _add_user_word(uid, pid)

    r = await skip_word(pid, uid)
    assert r["status"] == "skipped"
    assert not await _has_user_word(uid, pid)                     # убрано из учёбы юзера
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT COUNT(*) c FROM user_word_skips WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            assert (await cur.fetchone())["c"] == 1               # в личной свалке (больше не предложат)
        async with dbc.execute("SELECT COUNT(*) c FROM dict_words WHERE pool_id=? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id=?)", (pid, uid)) as cur:
            assert (await cur.fetchone())["c"] == 0               # отвязано от словарей
    finally:
        await _release(dbc)
    # НИКАКОЙ модерации: жалоб нет, слово на месте в общей базе
    assert await reported_count() == 0
    assert pid not in [w["pool_id"] for w in await reported_words()]
    assert "neaktuelt" in [w["norwegian"] for w in await pool_by_freq(100)]
