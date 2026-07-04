"""ЭТАП 9: реестр фоновых задач (db/bg_tasks.py).

Голый asyncio.create_task без ссылки GC мог убить на середине, а исключение —
в тишину. spawn держит ссылку до конца и логирует падение.
"""
import asyncio

from db import bg_tasks


async def test_spawn_completes_and_drops_ref():
    done = []

    async def work():
        await asyncio.sleep(0)
        done.append(1)

    t = bg_tasks.spawn(work(), name="ok")
    assert bg_tasks.pending_count() >= 1        # ссылка держится, пока в полёте
    await t
    await asyncio.sleep(0)                       # дать done-callback снять ссылку
    assert done == [1]
    assert t not in bg_tasks._TASKS


async def test_spawn_logs_exception_not_silent(caplog):
    async def boom():
        raise RuntimeError("падение фоновой задачи")

    with caplog.at_level("WARNING"):
        t = bg_tasks.spawn(boom(), name="cloze:1:2")
        await asyncio.gather(t, return_exceptions=True)
        await asyncio.sleep(0)
    assert any("cloze:1:2" in r.message and "упала" in r.message for r in caplog.records)
    assert t not in bg_tasks._TASKS             # ссылка снята даже после падения
