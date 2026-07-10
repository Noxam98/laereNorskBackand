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
async def test_resolve_compound_bank_wins_over_llm(fresh_db, monkeypatch):
    """Единый резолвер: банк — авторитет, data.compound — фолбэк только когда банк молчит."""
    from db import ordbank
    from db.pool import resolve_compound
    llm = {"compound": {"forledd": "xxx", "fuge": "", "etterledd": "yyy"}}

    # банк знает слово → LLM-разбор игнорируем
    monkeypatch.setattr(ordbank, "compound", lambda w: {"forledd": "barn", "fuge": "e", "etterledd": "hage",
                                                        "marked": None, "parts": ["barn", "hage"]})
    assert resolve_compound("barnehage", llm)["forledd"] == "barn"

    # банк не знает → берём data.compound (форма ответа та же: parts/marked присутствуют)
    monkeypatch.setattr(ordbank, "compound", lambda w: None)
    r = resolve_compound("xxxyyy", llm)
    assert r["forledd"] == "xxx" and r["etterledd"] == "yyy" and r["parts"] == ["xxx", "yyy"]

    # ни там, ни там → None
    assert resolve_compound("universitet", {}) is None
    assert resolve_compound("universitet", None) is None


@pytest.mark.asyncio
async def test_session_flashcard_gets_llm_compound(fresh_db):
    """Флешкарта-знакомство показывает разбор и для слов ВНЕ банка (ручной LLM-разбор).
    Раньше learning.py звал ordbank.compound() напрямую и такие слова не видел."""
    from db.learning import build_session
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "holdbarhetstid", ru="срок годности", pos="noun")
    await set_pool_compound(pid, {"forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"})
    res = await build_session(uid, size=20, lang="ru")
    el = next((e for e in res["words"] if e["pool_id"] == pid and e.get("mode") == "study"), None)
    assert el, "слово не попало на карточку-знакомство"
    assert el["compound"]["parts"] == ["holdbarhet", "tid"]


@pytest.mark.asyncio
async def test_pool_batch_after_carries_data(fresh_db):
    """compound_index_loop получает data — иначе резолвер видел бы только банк."""
    from db.compound_index import pool_batch_after
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "holdbarhetstid")
    await set_pool_compound(pid, {"forledd": "holdbarhet", "fuge": "s", "etterledd": "tid"})
    batch = await pool_batch_after(0, 300)
    row = next(r for r in batch if r[0] == pid)
    assert len(row) == 3 and row[1] == "holdbarhetstid"
    assert row[2]["compound"]["etterledd"] == "tid"   # data распарсена


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
