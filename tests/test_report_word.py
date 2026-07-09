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

    # 1) первая жалоба (юзер uid) → в очередь админа, и слово убрано из Учёбы пользователя
    r = await report_word(pid, uid)
    assert r["status"] == "queued"
    assert not await _has_user_word(uid, pid)
    assert await reported_count() == 1
    assert [w["pool_id"] for w in await reported_words()] == [pid]

    # дедуп: повторная жалоба ТОГО ЖЕ юзера не крутит счётчик и не будит модератора повторно
    assert (await report_word(pid, uid))["status"] == "already"
    assert await reported_count() == 1

    # 2) админ «оставить» → жалоба снята, выбор запомнен на следующие 5 жалоб
    await resolve_report(pid, "keep")
    assert await reported_count() == 0

    # 3) следующие 5 жалоб от РАЗНЫХ юзеров гасятся автоматически (не идут в очередь).
    # Один юзер = одна жалоба (дедуп), поэтому реалистичный сценарий гашения — разные юзеры.
    for i in range(5):
        u2, _ = await pytest.seed_user(f"rep{i}")
        assert (await report_word(pid, u2))["status"] == "dismissed"
    assert await reported_count() == 0

    # 4) 6-я жалоба нового юзера (память исчерпана) → снова в очередь админа
    u3, _ = await pytest.seed_user("rep6")
    assert (await report_word(pid, u3))["status"] == "queued"
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
async def test_known_goes_to_known_bucket_not_progress(fresh_db):
    """«Уже знаю» → корзина «Знаю» (status 'known'): НЕ в «Выучено», НЕ архив, прогресс уровня не двигает."""
    from db import learning_set_status, learning_stats
    from db.learning import build_session
    uid, did = await pytest.seed_user()
    pid, _ = await pytest.seed_word(did, "kjentord")
    before = await learning_stats(uid)
    await learning_set_status(uid, pid, "known")
    st = await learning_stats(uid)
    assert st["byStatus"].get("known", 0) == 1          # в корзине «Знаю»
    assert st["byStatus"].get("mastered", 0) == 0       # НЕ выучено
    assert st["byStatus"].get("archived", 0) == 0       # и не архив
    assert st["toNextLevel"] == before["toNextLevel"]   # прогресс уровня не сдвинулся
    # знакомое слово больше не попадает в сессию
    res = await build_session(uid, size=20)
    assert all(w.get("pool_id") != pid for w in res["words"])


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
