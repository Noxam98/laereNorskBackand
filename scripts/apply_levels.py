#!/usr/bin/env python3
"""НА СЕРВЕРЕ: применяет data/levels-v1.json (наш открытый словник уровней) к word_pool.
Ключ — (norwegian, pos); слова вне словника не трогаются."""
import json
import sqlite3
import sys

levels = json.load(open(sys.argv[1]))["levels"]
c = sqlite3.connect("/opt/norsk/data/users.db", timeout=30)
changed = same = 0
for wid, no, pos, cur in c.execute(
        "SELECT id, norwegian, pos, level FROM word_pool WHERE pos IS NOT NULL").fetchall():
    lvl = levels.get(f"{(no or '').lower()}|{pos}")
    if not lvl:
        continue
    if lvl == cur:
        same += 1
        continue
    c.execute("UPDATE word_pool SET level=? WHERE id=?", (lvl, wid))
    changed += 1
c.commit()
dist = dict(c.execute("SELECT level, COUNT(*) FROM word_pool GROUP BY level ORDER BY level"))
print(f"уровни: совпало {same} · обновлено {changed}")
print("распределение:", {k: dist.get(k, 0) for k in ("A1", "A2", "B1", "B2", "C1", "C2")})
