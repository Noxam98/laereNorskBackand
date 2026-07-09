"""Хард-бонд пула соединений (db.core): ограничиваем число ОДНОВРЕМЕННЫХ соединений, но НЕ дедлочим
на вложенных _conn (функция под соединением зовёт хелпер со своим _conn). Ключевое свойство —
вложенный _conn не занимает слот и не блокируется, даже когда бонд исчерпан.

Каждый «пользовательский» захват моделируем задачей со СВЕЖИМ контекстом (contextvars.Context) —
как реальный запрос-хендлер: его _conn внешний (глубина 0) и честно берёт слот."""
import asyncio
import contextvars

import db.core as core
from db.core import _conn, _release


async def _hold(entered: asyncio.Future, release_ev: asyncio.Event, out: list):
    """Взять внешнее соединение, сигналить о захвате, дождаться отмашки, отпустить."""
    c = await _conn()
    out.append(c)
    if not entered.done():
        entered.set_result(None)
    await release_ev.wait()
    await _release(c)


def _spawn_fresh(coro):
    """Задача со свежим контекстом → её первый _conn считается внешним (как запрос-хендлер),
    а не унаследованно-вложенным от родителя, который сейчас держит соединение."""
    return asyncio.create_task(coro, context=contextvars.Context())


async def test_nested_conn_does_not_block_on_exhausted_bound(fresh_db):
    """Гвоздь дизайна: держим единственный слот внешним _conn, затем ВЛОЖЕННЫЙ _conn в той же
    задаче обязан выдать соединение немедленно (иначе — вечный дедлок вложенного захвата)."""
    core._POOL_SEM = asyncio.Semaphore(1)
    a = await _conn()                                  # внешний: занял единственный слот
    try:
        assert core._POOL_SEM._value == 0              # бонд исчерпан
        b = await asyncio.wait_for(_conn(), 2)         # вложенный: НЕ должен блокироваться
        assert b is not None
        await _release(b)
        assert core._POOL_SEM._value == 0              # release вложенного не вернул внешний слот
    finally:
        await _release(a)
    assert core._POOL_SEM._value == 1                  # release внешнего вернул слот


async def test_pool_bounds_concurrent_outer_conns(fresh_db):
    """Внешних захватов не больше бонда: 3-й ждёт, пока кто-то не отпустит слот."""
    core._POOL_SEM = asyncio.Semaphore(2)
    loop = asyncio.get_running_loop()
    release_ev = asyncio.Event()
    out = []
    e1, e2, e3 = loop.create_future(), loop.create_future(), loop.create_future()
    t1 = _spawn_fresh(_hold(e1, release_ev, out))
    t2 = _spawn_fresh(_hold(e2, release_ev, out))
    await asyncio.wait_for(asyncio.gather(e1, e2), 2)  # оба взяли слот
    assert core._POOL_SEM._value == 0

    t3 = _spawn_fresh(_hold(e3, release_ev, out))
    await asyncio.sleep(0.05)
    assert not e3.done()                               # 3-й заблокирован — слотов нет
    release_ev.set()                                   # отпускаем всех
    await asyncio.wait_for(asyncio.gather(t1, t2, t3), 3)
    assert len(out) == 3
    assert core._POOL_SEM._value == 2                  # все слоты вернулись


async def test_slot_not_leaked_across_cycles(fresh_db):
    """Много пар _conn/_release не подтравливают слоты — бонд остаётся целым."""
    core._POOL_SEM = asyncio.Semaphore(1)
    for _ in range(25):
        c = await _conn()
        await _release(c)
    assert core._POOL_SEM._value == 1                  # permit полностью возвращён
    c = await asyncio.wait_for(_conn(), 1)             # и соединение всё ещё выдаётся сразу
    await _release(c)


async def test_slot_released_when_make_conn_fails(fresh_db, monkeypatch):
    """Ошибка выдачи соединения не подвешивает слот навсегда (permit возвращается на исключении)."""
    core._POOL_SEM = asyncio.Semaphore(1)

    async def _boom():
        raise RuntimeError("no conn")

    monkeypatch.setattr(core, "_make_conn", _boom)
    try:
        await _conn()
        assert False, "ожидали RuntimeError"
    except RuntimeError:
        pass
    assert core._POOL_SEM._value == 1                  # слот вернулся, не утёк
