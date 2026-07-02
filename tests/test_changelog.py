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
    entries = await get_changelog()
    assert len(entries) == 2
    assert entries[0]["kind"] == "fix"                      # свежие первыми (вторая запись — последняя)
    assert entries[1]["i18n"]["ru"]["t"] == "Трек форм"
    assert entries[0]["day"] and entries[0]["id"] > entries[1]["id"]


async def test_idempotent_by_source(fresh_db):
    await add_changelog("frontend", "abc..def", _E)
    r2 = await add_changelog("frontend", "abc..def", _E)   # повторный прогон хука
    assert r2 == {"added": 0, "skipped": True}
    assert len(await get_changelog()) == 2
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
    entries = await get_changelog()
    assert entries[0]["kind"] == "feature"
