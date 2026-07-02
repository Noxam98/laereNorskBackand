"""Ченжлог: приём пачки от стража пуша (идемпотентность по (repo, source)) + выдача."""
from db.changelog import add_changelog, get_changelog

_E = [
    {"kind": "feature", "i18n": {"ru": {"t": "Трек форм", "d": "Учим формы слов"},
                                 "en": {"t": "Forms track", "d": "Learn word forms"}}},
    {"kind": "fix", "i18n": {"ru": {"t": "Фикс скролла", "d": "Мобильный скролл"}}},
]


async def test_add_and_get(fresh_db):
    r = await add_changelog("frontend", "abc123..def456", _E)
    assert r == {"added": 2, "skipped": False}
    page = await get_changelog()
    entries = page["entries"]
    assert len(entries) == 2 and page["total"] == 2
    assert entries[0]["kind"] == "fix"                      # свежие первыми (вторая запись — последняя)
    assert entries[1]["i18n"]["ru"]["t"] == "Трек форм"
    assert entries[0]["day"] and entries[0]["id"] > entries[1]["id"]


async def test_idempotent_by_source(fresh_db):
    await add_changelog("frontend", "abc..def", _E)
    r2 = await add_changelog("frontend", "abc..def", _E)   # повторный прогон хука
    assert r2 == {"added": 0, "skipped": True}
    assert (await get_changelog())["total"] == 2
    # другой range того же репо — добавляется
    r3 = await add_changelog("frontend", "def..fff", _E[:1])
    assert r3["added"] == 1


async def test_validation_skips_junk(fresh_db):
    r = await add_changelog("backend", "x..y", [
        {"kind": "weird", "i18n": {"ru": {"t": "Ок", "d": ""}}},   # неизвестный kind → feature
        {"kind": "fix", "i18n": {}},                                # пустой i18n → пропуск
        {"kind": "fix", "i18n": {"ru": {"t": "", "d": "без заголовка"}}},  # нет заголовка → пропуск
    ])
    assert r["added"] == 1
    assert (await get_changelog())["entries"][0]["kind"] == "feature"


async def test_pagination_offset(fresh_db):
    # 5 записей → страница по 2 с offset листает всю историю, total честный
    for i in range(5):
        await add_changelog("frontend", f"r{i}..r{i+1}",
                            [{"kind": "fix", "i18n": {"ru": {"t": f"Запись {i}", "d": ""}}}])
    p1 = await get_changelog(limit=2, offset=0)
    p2 = await get_changelog(limit=2, offset=2)
    p3 = await get_changelog(limit=2, offset=4)
    assert p1["total"] == 5 and len(p1["entries"]) == 2 and len(p3["entries"]) == 1
    ids = [e["id"] for e in p1["entries"] + p2["entries"] + p3["entries"]]
    assert ids == sorted(ids, reverse=True) and len(set(ids)) == 5
