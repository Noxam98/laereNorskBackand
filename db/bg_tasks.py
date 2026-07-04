"""Мини-реестр фоновых задач (Этап 9).

`asyncio.create_task(coro)` без сохранённой ссылки — известная мина: сборщик мусора
вправе убить задачу на середине (в docs asyncio прямо просят держать strong-ref), а
необработанное исключение в fire-and-forget уходит в тишину (видно лишь на закрытии
цикла). spawn() держит ссылку до завершения и логирует падение.
"""
import asyncio
import logging

logger = logging.getLogger("norsk.bg")

_TASKS: set[asyncio.Task] = set()


def spawn(coro, *, name: str) -> asyncio.Task:
    """Запустить фоновую корутину с сохранением ссылки и логированием исключений.
    name — для лога (что именно упало)."""
    task = asyncio.ensure_future(coro)
    _TASKS.add(task)

    def _done(t: asyncio.Task):
        _TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.warning("фоновая задача %s упала: %r", name, exc)

    task.add_done_callback(_done)
    return task


def pending_count() -> int:
    """Сколько фоновых задач сейчас в полёте (для тестов/диагностики)."""
    return len(_TASKS)
