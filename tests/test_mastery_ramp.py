"""Мастери-рампа из 4 клеток: choice_no2int → choice_int2no → build_int2no → input_int2no.
Каждая клетка хранит '1' (пройдена) / '' (сброс ошибкой); mastered = все 4 == '1'."""
import pytest
from db.learning import apply_result, REQUIRED_CELLS, _mastered_by_modes, status_of
from tests.conftest import seed_user, seed_word


async def _master_all(uid, pid):
    """Пройти все 4 клетки рампы верными ответами."""
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    await apply_result(uid, pid, True, mode="build", direction="int2no")
    return await apply_result(uid, pid, True, mode="input", direction="int2no")


async def test_full_ramp_masters(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    r = await _master_all(uid, pid)
    assert r["mastered"] is True
    assert all(r["modes"][c] == "1" for c in REQUIRED_CELLS)
    assert status_of(
        {"correct": 4, "incorrect": 0, "strength": r["strength"], "reps": 4}, r["modes"]
    ) == "mastered"


async def test_three_cells_not_mastered(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    await apply_result(uid, pid, True, mode="choice", direction="no2int")
    await apply_result(uid, pid, True, mode="choice", direction="int2no")
    r = await apply_result(uid, pid, True, mode="build", direction="int2no")
    assert r["mastered"] is False
    assert _mastered_by_modes(r["modes"]) is False


async def test_error_resets_only_that_cell(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "katt", "кот")
    await _master_all(uid, pid)
    # ошибка в input — сбрасывает только эту клетку
    r = await apply_result(uid, pid, False, mode="input", direction="int2no")
    assert r["modes"]["input_int2no"] == ""
    assert r["modes"]["choice_no2int"] == "1"
    assert r["mastered"] is False
    # повторно верно — снова mastered
    r = await apply_result(uid, pid, True, mode="input", direction="int2no")
    assert r["mastered"] is True


async def test_study_is_passive(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bok", "книга")
    r = await apply_result(uid, pid, True, mode="study", direction="no2int")
    assert all(c not in r["modes"] for c in REQUIRED_CELLS)
    assert r["modes"]["hist"] == "1"   # окно силы всё равно обновлено


async def test_build_no2int_ignored(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "by", "город")
    r = await apply_result(uid, pid, True, mode="build", direction="no2int")
    assert "build_no2int" not in r["modes"]
    assert "build_int2no" not in r["modes"]
    assert r["modes"]["hist"] == "1"   # счётчики/окно обновлены как раньше


async def test_direction_none_backward_compat(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "vann", "вода")
    r = await apply_result(uid, pid, True, elapsed=2.0, mode="choice")
    assert all(c not in r["modes"] for c in REQUIRED_CELLS)
    assert r["modes"]["hist"] == "1"
    assert r["strength"] >= 0
