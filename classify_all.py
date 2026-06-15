#!/usr/bin/env python3
"""Разовая пакетная классификация пула: проставить уровень CEFR + темы словам без меты.
Идемпотентно/возобновляемо — берёт только level IS NULL. Пачками по CLASSIFY_BATCH.

Запуск (на сервере):  python classify_all.py            # все непроклассифицированные
                       python classify_all.py --limit 200 --sleep 3
"""
import argparse
import asyncio

from db import init_db, pool_missing_meta
from main import classify_batch, CLASSIFY_BATCH


async def run(limit: int, batch: int, sleep: float):
    await init_db()
    total = 0
    while True:
        if limit and total >= limit:
            break
        take = min(batch, limit - total) if limit else batch
        items = await pool_missing_meta(take)
        if not items:
            print(f"готово: больше непроклассифицированных слов нет. всего: {total}")
            break
        done = await classify_batch(items)
        total += len(items)
        print(f"пачка: {done}/{len(items)} проставлено (накоплено обработано: {total})", flush=True)
        if done == 0:
            # вероятно квота/ошибка — не молотим вхолостую
            print("пачка без результата (квота/ошибка?) — стоп.")
            break
        await asyncio.sleep(sleep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="макс. слов за прогон (0 = все)")
    ap.add_argument("--batch", type=int, default=CLASSIFY_BATCH, help="слов на один LLM-вызов")
    ap.add_argument("--sleep", type=float, default=2.0, help="пауза между пачками, сек")
    a = ap.parse_args()
    asyncio.run(run(a.limit, a.batch, a.sleep))
