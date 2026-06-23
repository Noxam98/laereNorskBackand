"""Движок сессии build_session: программу занятия ведёт сервер (игрок режим не выбирает).
Проверяем реальное поведение приоритизации и пейсинга, а не подгонку под реализацию:
  (1) свежее слово (0 попыток) → первый шаг 'card' / клетка choice_no2int;
  (2) длина сессии не превышает size;
  (3) лимит одновременно-в-работе: при >=WIP_LIMIT не-mastered новые не подсыпаются;
  (4) просроченные по due идут раньше дозревающих (new/learning).
"""
import pytest
from db.core import _conn, _release, _now
from db.learning import (
    build_session, apply_result, REQUIRED_CELLS, WIP_LIMIT,
    status_of, _is_due, _due_str,
)
from tests.conftest import seed_user, seed_word


async def _set_due(pool_id, user_id, due_at):
    """Подкрутить due_at у строки слова напрямую (моделируем просрочку/дозревание)."""
    db = await _conn()
    try:
        await db.execute(
            "UPDATE user_words SET due_at = ? WHERE user_id = ? AND pool_id = ?",
            (due_at, user_id, pool_id))
        await db.commit()
    finally:
        await _release(db)


async def _age_seen(pool_id, user_id):
    """Состарить last_seen (за кулдаун) — слово перестаёт считаться «только что показанным»,
    иначе «умная очередь» отправит его в хвост и собьёт проверку приоритета."""
    db = await _conn()
    try:
        await db.execute(
            "UPDATE user_words SET last_seen = ? WHERE user_id = ? AND pool_id = ?",
            ("2000-01-01T00:00:00", user_id, pool_id))
        await db.commit()
    finally:
        await _release(db)


# (1) свежее слово → первый шаг — пассивная карточка, первая клетка рампы
async def test_fresh_word_first_step_is_card(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    res = await build_session(uid, size=20)
    words = res["words"]
    assert len(words) == 1
    w = words[0]
    assert w["pool_id"] == pid
    # совсем новое (0 попыток) → пассивная карточка study, без направления
    assert w["mode"] == "study"
    assert w["step"] == "card"
    assert w["direction"] is None


async def test_first_test_cell_after_card(fresh_db):
    """После прохождения карточки (любой ответ создаёт строку) первая клетка рампы —
    choice_no2int (первая невыполненная ступень REQUIRED_CELLS)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "bil", "машина")
    # «прошли» карточку — даём один ответ по первой клетке неверно, чтобы попытки>0,
    # но клетка осталась не пройдена → следующий шаг снова первая клетка рампы
    await apply_result(uid, pid, False, mode="choice", direction="no2int")
    res = await build_session(uid, size=20)
    w = res["words"][0]
    assert w["pool_id"] == pid
    assert w["step"] == REQUIRED_CELLS[0] == "choice_no2int"
    assert w["mode"] == "choice"
    assert w["direction"] == "no2int"


# (2) длина сессии не превышает size
async def test_session_not_longer_than_size(fresh_db):
    uid, did = await seed_user()
    for i in range(15):
        await seed_word(did, f"ord{i}", f"слово{i}")
    res = await build_session(uid, size=5)
    assert len(res["words"]) == 5
    # все уникальны
    ids = [w["pool_id"] for w in res["words"]]
    assert len(set(ids)) == 5


async def test_session_size_caps_at_available(fresh_db):
    """Если слов меньше size — отдаём столько, сколько есть, без дублей."""
    uid, did = await seed_user()
    for i in range(3):
        await seed_word(did, f"w{i}", f"п{i}")
    res = await build_session(uid, size=20)
    assert len(res["words"]) == 3


# (3) лимит in-progress: при >=WIP_LIMIT не-mastered новые не добавляются
async def test_wip_limit_blocks_new_words(fresh_db):
    uid, did = await seed_user()
    # WIP_LIMIT слов делаем «в работе» (learning/review): один верный ответ → reps=1, не mastered
    in_work_pids = []
    for i in range(WIP_LIMIT):
        pid, _ = await seed_word(did, f"work{i}", f"раб{i}")
        await apply_result(uid, pid, True, mode="choice", direction="no2int")
        in_work_pids.append(pid)
    # + несколько совсем новых (0 попыток)
    new_pids = []
    for i in range(5):
        pid, _ = await seed_word(did, f"new{i}", f"нов{i}")
        new_pids.append(pid)

    # запас «в работе» должен набраться именно на лимите
    res = await build_session(uid, size=50)
    picked = {w["pool_id"] for w in res["words"]}
    # ни одно совсем новое слово не должно попасть — лимит исчерпан словами в работе
    assert picked.isdisjoint(set(new_pids))
    # а слова в работе — попадают (их и набираем для дозревания)
    assert picked.issubset(set(in_work_pids))


async def test_new_words_flow_when_under_wip_limit(fresh_db):
    """Под лимитом новые слова подсыпаются (контроль к предыдущему тесту)."""
    uid, did = await seed_user()
    # один в работе — лимит далеко не исчерпан
    pid_work, _ = await seed_word(did, "work", "раб")
    await apply_result(uid, pid_work, True, mode="choice", direction="no2int")
    new_pids = []
    for i in range(3):
        pid, _ = await seed_word(did, f"new{i}", f"нов{i}")
        new_pids.append(pid)
    res = await build_session(uid, size=50)
    picked = {w["pool_id"] for w in res["words"]}
    # новые слова доступны, раз лимит не достигнут
    assert set(new_pids).issubset(picked)


# (4) просроченные идут раньше дозревающих
async def test_overdue_before_maturing(fresh_db):
    uid, did = await seed_user()
    # слово A: ответили верно (стало review/в работе), затем делаем его просроченным
    pid_over, _ = await seed_word(did, "overdue", "просрочено")
    await apply_result(uid, pid_over, True, mode="choice", direction="no2int")
    await _set_due(pid_over, uid, _due_str(-2))   # due в прошлом → просрочено

    # слово B: ответили верно, due далеко в будущем → дозревает, но не просрочено
    pid_mat, _ = await seed_word(did, "maturing", "дозревает")
    await apply_result(uid, pid_mat, True, mode="choice", direction="int2no")
    await _set_due(pid_mat, uid, _due_str(30))

    res = await build_session(uid, size=20)
    order = [w["pool_id"] for w in res["words"]]
    assert pid_over in order and pid_mat in order
    # просроченное раньше дозревающего
    assert order.index(pid_over) < order.index(pid_mat)


async def test_overdue_before_fresh_new(fresh_db):
    """Просроченное обгоняет совсем новые (0 попыток) слова в очереди."""
    uid, did = await seed_user()
    pid_over, _ = await seed_word(did, "overdue", "просрочено")
    await apply_result(uid, pid_over, True, mode="choice", direction="no2int")
    await _set_due(pid_over, uid, _due_str(-5))
    await _age_seen(pid_over, uid)   # просрочка = давно не виделись (иначе кулдаун уведёт в хвост)
    pid_new, _ = await seed_word(did, "brandnew", "новое")

    res = await build_session(uid, size=20)
    order = [w["pool_id"] for w in res["words"]]
    assert order.index(pid_over) < order.index(pid_new)
