"""Слуховая сессия: при включённых аудиозаданиях аудио-клетка (choice_no2int) ОТКЛАДЫВАЕТСЯ из
дневной рампы в отдельную слуховую партию; сдача на слух закрывает 4-ю клетку → слово полностью
выучено. Ошибка на слух не откатывает текстовые ступени. При ВЫКЛ — choice_no2int идёт в дневной
рампе (текстом), слуховой сессии нет."""
import json
from db.core import _conn, _release
from db.learning import (
    build_session, build_listen_session, listen_status, apply_result, REQUIRED_CELLS, _AUDIO_CELL,
)
from db.users import set_user_game_prefs
from tests.conftest import seed_user, seed_word

_TEXT = [("choice", "int2no"), ("build", "int2no"), ("input", "int2no")]   # 3 текстовые клетки рампы


async def _text_master(uid, pid):
    """Пройти 3 текстовые клетки (choice_int2no/build_int2no/input_int2no) — слово «ждёт слух»."""
    for mode, direction in _TEXT:
        await apply_result(uid, pid, True, mode=mode, direction=direction)


async def _modes(uid, pid):
    db = await _conn()
    try:
        async with db.execute("SELECT modes FROM user_words WHERE user_id=? AND pool_id=?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return json.loads(r["modes"]) if r and r["modes"] else {}


async def test_audio_deferred_from_daily(fresh_db):
    """Аудио ВКЛ (дефолт): слово с 3 текстовыми клетками НЕ предлагается в дневной сессии (ждёт слух)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    res = await build_session(uid, size=20)
    assert pid not in {w["pool_id"] for w in res["words"]}
    st = await listen_status(uid)
    assert st["audio"] is True and st["pending"] == 1


async def test_listen_session_surfaces_word(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    res = await build_listen_session(uid, size=20)
    ws = res["words"]
    assert len(ws) == 1 and ws[0]["pool_id"] == pid
    assert ws[0]["step"] == _AUDIO_CELL and ws[0]["mode"] == "choice" and ws[0]["direction"] == "no2int"
    assert ws[0]["listen"] is True and res["composition"]["listen"] == 1


async def test_audio_pass_completes_mastery(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    assert not (await _modes(uid, pid)).get(_AUDIO_CELL)             # ещё не сдано на слух
    await apply_result(uid, pid, True, mode="choice", direction="no2int")   # сдал слух
    m = await _modes(uid, pid)
    assert all(m.get(c) == "1" for c in REQUIRED_CELLS)             # все 4 клетки → выучено
    assert (await listen_status(uid))["pending"] == 0              # больше не ждёт слух


async def test_audio_error_no_rollback(fresh_db):
    """Ошибка на слух НЕ откатывает текстовые ступени — слово остаётся «ждёт слух»."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    await apply_result(uid, pid, False, mode="choice", direction="no2int")
    m = await _modes(uid, pid)
    assert all(m.get(c) == "1" for c in REQUIRED_CELLS if c != _AUDIO_CELL)   # текст цел
    assert m.get(_AUDIO_CELL) != "1"
    assert (await listen_status(uid))["pending"] == 1


async def test_audio_off_keeps_cell_in_daily(fresh_db):
    """Аудио ВЫКЛ: choice_no2int идёт в дневной рампе (текстом) — не откладывается, слуховой нет."""
    uid, did = await seed_user()
    await set_user_game_prefs(uid, json.dumps({"audio": False}))
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    res = await build_session(uid, size=20)
    w = [x for x in res["words"] if x["pool_id"] == pid]
    assert w and w[0]["step"] == _AUDIO_CELL                        # choice_no2int предлагается в дневной
    assert (await listen_status(uid))["audio"] is False            # слуховой сессии нет


async def test_audio_pending_due_not_double_shown(fresh_db):
    """Слово «ждёт слух», ставшее due, НЕ показывается в дневной сессии повтором — только в слуховой.
    (audio-pending ≠ mastered → не в due_mastered; _next_step(audio_on) → None → не в дневной.)"""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await _text_master(uid, pid)
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET due_at = ? WHERE pool_id = ?", ("2000-01-01T00:00:00", pid))
        await db.commit()
    finally:
        await _release(db)
    daily = await build_session(uid, size=20)
    listen = await build_listen_session(uid, size=20)
    assert pid not in {w["pool_id"] for w in daily["words"]}      # не в дневной даже будучи due
    assert pid in {w["pool_id"] for w in listen["words"]}         # только в слуховой


async def test_listen_threshold_pack(fresh_db):
    """Порог партии listenPack (5..20): ready=True когда pending >= порог."""
    uid, did = await seed_user()
    await set_user_game_prefs(uid, json.dumps({"listenPack": 5}))
    for i in range(5):
        pid, _ = await seed_word(did, f"ord{i}", f"сл{i}")
        await _text_master(uid, pid)
    st = await listen_status(uid)
    assert st["pack"] == 5 and st["pending"] == 5 and st["ready"] is True
