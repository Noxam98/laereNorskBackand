"""Ченжлог приложения: хранение и выдача записей «что нового».

Записи создаёт страж пуша (git pre-push хук → Claude Sonnet суммирует диффы в юзерские
фичи/фиксы → POST /admin/changelog). Идемпотентность — по (repo, source): повторный прогон
хука по тому же git-range не плодит дублей. Выдача — свежие первыми, i18n целиком
(фронт сам выбирает язык с фолбэком)."""
import json

from .core import _conn, _release, _now

KINDS = ("feature", "fix", "perf", "ui")


async def add_changelog(repo, source, entries):
    """Добавить пачку записей одного пуша. entries: [{kind, i18n:{lang:{t,d}}}].
    Возвращает {added, skipped:bool}. skipped=True — этот (repo, source) уже загружен."""
    db = await _conn()
    try:
        if source:
            async with db.execute("SELECT 1 FROM changelog WHERE repo = ? AND source = ? LIMIT 1",
                                  (repo, source)) as cur:
                if await cur.fetchone():
                    return {"added": 0, "skipped": True}
        now = _now()
        day = now[:10]
        added = 0
        for e in entries:
            kind = (e.get("kind") or "").strip().lower()
            i18n = e.get("i18n") or {}
            # минимальная валидация: известный тип + есть хоть один язык с заголовком
            if kind not in KINDS:
                kind = "feature"
            if not any(isinstance(v, dict) and (v.get("t") or "").strip() for v in i18n.values()):
                continue
            await db.execute(
                "INSERT INTO changelog (day, repo, source, kind, i18n, created_at) VALUES (?,?,?,?,?,?)",
                (day, repo, source, kind, json.dumps(i18n, ensure_ascii=False), now))
            added += 1
        await db.commit()
        return {"added": added, "skipped": False}
    finally:
        await _release(db)


async def get_changelog(limit=30):
    """Свежие записи (новые первыми): [{id, day, kind, i18n}]."""
    db = await _conn()
    try:
        async with db.execute(
                "SELECT id, day, kind, i18n FROM changelog ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit), 100)),)) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            try:
                i18n = json.loads(r["i18n"])
            except Exception:
                continue
            out.append({"id": r["id"], "day": r["day"], "kind": r["kind"], "i18n": i18n})
        return out
    finally:
        await _release(db)
