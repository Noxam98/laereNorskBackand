"""Ручной разбор составного слова из карточки (для слов вне ordbank leddanalyse):
персист в data.compound + обратный индекс, «не составное» помечается проверенным,
и валидация LLM-разбора (складывается в слово + части — известные леммы)."""
import pytest

from db.core import _conn, _release
from db import get_pool_meta, set_pool_compound
from tests.conftest import seed_user, seed_word


@pytest.mark.asyncio
async def test_set_pool_compound_persists_and_meta_exposes(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "holdbarhetstid", ru="срок годности", pos="noun")
    await set_pool_compound(pid, {"forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"})
    m = await get_pool_meta("holdbarhetstid", user_id=uid)
    # резолвер добавляет marked/parts при фолбэке на data.compound — проверяем ключевые поля
    assert {k: m["compound"][k] for k in ("forledd", "fuge", "etterledd")} == \
        {"forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"}
    assert m["compoundChecked"] is True
    # обратный индекс частей получил строку (разблокировка композитов по выученным основам)
    db = await _conn()
    try:
        async with db.execute("SELECT forledd, etterledd FROM word_pool_compounds WHERE pool_id=?", (pid,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    assert row and row["forledd"] == "holdbarhet" and row["etterledd"] == "tid"


@pytest.mark.asyncio
async def test_set_pool_compound_none_marks_checked(fresh_db):
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "universitet", pos="noun")   # заимствование — не составное
    await set_pool_compound(pid, None)
    m = await get_pool_meta("universitet", user_id=uid)
    assert m["compound"] is None
    assert m["compoundChecked"] is True   # проверено → пункт меню на фронте исчезнет


@pytest.mark.asyncio
async def test_analyze_compound_validation(fresh_db, monkeypatch):
    # ordbank.db в тестах нет → is_lemma=False, поэтому «известность» частей обеспечиваем пулом
    uid, did = await seed_user()
    await seed_word(did, "holdbarhet"); await seed_word(did, "tid")
    import routers.pool as P

    def ask(res):
        async def _f(*a, **k):
            return res
        return _f

    # валидный разбор: складывается в слово, обе части — в пуле
    monkeypatch.setattr(P, "ask_json", ask({"is_compound": True, "forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"}))
    assert await P._analyze_compound("holdbarhetstid") == {"forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"}

    # части НЕ складываются в исходное слово буква-в-букву → отклоняем
    monkeypatch.setattr(P, "ask_json", ask({"is_compound": True, "forledd": "holdbar", "fuge": "", "etterledd": "tid"}))
    assert await P._analyze_compound("holdbarhetstid") is None

    # часть неизвестна (нет ни в банке, ни в пуле) → отклоняем (антигаллюцинация)
    monkeypatch.setattr(P, "ask_json", ask({"is_compound": True, "forledd": "xyzzy", "fuge": "", "etterledd": "qwerty"}))
    assert await P._analyze_compound("xyzzyqwerty") is None

    # LLM говорит «не составное» → None
    monkeypatch.setattr(P, "ask_json", ask({"is_compound": False}))
    assert await P._analyze_compound("universitet") is None
