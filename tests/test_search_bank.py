"""Поиск по словоформам через банк (gikk→gå), добор автокомплита из банка,
и добавление слова путём банк+Lexin (LLM не трогается — Lexin замокан)."""
import json
import sqlite3

import pytest

import db as D
from db import ordbank
from db.core import _conn


@pytest.fixture
def bank(tmp_path, monkeypatch):
    """Мини-банк: gå/verb (форма gikk), hus/noun (форма huset), лемма bok/noun."""
    p = tmp_path / "ordbank.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE forms (norwegian TEXT, pos TEXT, forms TEXT, PRIMARY KEY (norwegian,pos))")
    c.execute("CREATE TABLE formindex (form TEXT, norwegian TEXT, pos TEXT, PRIMARY KEY (form,norwegian,pos))")
    c.execute("INSERT INTO forms VALUES ('gå','verb', ?)",
              (json.dumps({"pos": "verb", "present": "går", "past": "gikk", "perfect": "har gått"}),))
    c.execute("INSERT INTO forms VALUES ('bok','noun', ?)",
              (json.dumps({"pos": "noun", "gender": "ei", "def_sg": "boka"}),))
    for f in ("gå", "går", "gikk", "gått"):
        c.execute("INSERT INTO formindex VALUES (?,?,?)", (f, "gå", "verb"))
    c.execute("INSERT INTO formindex VALUES ('bok','bok','noun')")
    c.execute("INSERT INTO formindex VALUES ('bøker','bok','noun')")
    c.commit(); c.close()
    monkeypatch.setattr(ordbank, "PATH", str(p))
    monkeypatch.setattr(ordbank, "_conn", None)
    yield
    monkeypatch.setattr(ordbank, "_conn", None)


async def _seed_pos(no, pos, ru):
    dbc = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, created_at) VALUES (?,?,?,datetime('now'))",
            (no, data, pos))
        await dbc.commit()
        return cur.lastrowid
    finally:
        from db.core import _release
        await _release(dbc)


async def test_search_by_form_finds_lemma(fresh_db, bank):
    await _seed_pos("gå", "verb", "идти")
    res = await D.search_pool("gikk", limit=5)
    hit = next((r for r in res if r["word"] == "gå"), None)
    assert hit and hit["inPool"] and hit.get("viaForm") == "gikk"


async def test_search_bank_suggests_missing_lemma(fresh_db, bank):
    res = await D.search_pool("bok", limit=5)
    hit = next((r for r in res if r["word"] == "bok"), None)
    assert hit and hit["inPool"] is False and hit["part_of_speech"] == "noun"


async def test_generate_uses_bank_and_lexin(fresh_db, bank, monkeypatch):
    import lexin as lexin_live
    import routers.pool as RP

    async def fake_lookup(word, pos):
        assert (word, pos) == ("gå", "verb")     # форма gikk сведена к лемме
        return {"ru": ["идти"], "en": ["go"]}
    monkeypatch.setattr(lexin_live, "lookup", fake_lookup)

    out = await RP.pool_generate({"word": "gikk"}, user={"id": 1})
    assert out["generated"] and out["word"] == "gå" and out.get("viaForm") == "gikk"
    assert out["translate"]["ru"] == ["идти"]
    pid = out["pool_id"]
    w = await D.get_pool_by_id(pid)
    forms = w.get("forms") if isinstance(w, dict) else None
    assert forms and forms.get("past") == "gikk"   # формы легли из банка, не из LLM


async def test_grid_finds_word_by_form(fresh_db, bank):
    await _seed_pos("gå", "verb", "идти")
    res = await D.get_pool_list(limit=10, q="gikk")
    words = [w["word"] for w in res["words"]]
    assert "gå" in words   # грид Базы сводит форму к лемме через банк


async def _seed_learning_word(dbc, uid, did, no, modes, due_shift="+2 days"):
    """Слово в наборе юзера с прогрессом: рампа по modes, due в будущем."""
    cur = await dbc.execute(
        "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
        (no, json.dumps({"translate": {"no": [no], "ru": ["слово"]}, "part_of_speech": "noun"}), "noun", "A1"))
    pid = cur.lastrowid
    await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                      (did, pid))
    await dbc.execute(
        "INSERT INTO user_words (user_id, pool_id, strength, reps, correct, incorrect, "
        "interval_days, due_at, modes, created_at) "
        f"VALUES (?,?,?,?,?,?,?,datetime('now','{due_shift}'),?,datetime('now'))",
        (uid, pid, 3, 3, 3, 0, 2, modes))
    return pid


# рампа сдана вся, КРОМЕ аудио-клетки → слово «ждёт слух» (живёт в слуховой партии)
_AUDIO_PENDING = '{"choice_int2no":"1","build_int2no":"1","input_int2no":"1"}'


async def test_audio_pending_words_dont_choke_wip(fresh_db, bank):
    """Тупик ветерана 3.07: 21 слово ждёт слуховой сдачи (>WIP_LIMIT) + новые в наборе.
    Слуховые НЕ должны душить WIP — дневная сессия обязана отдать новые слова."""
    from tests.conftest import seed_user
    from db.learning import build_session
    uid, did = await seed_user("veteran")
    dbc = await _conn()
    try:
        for i in range(21):
            await _seed_learning_word(dbc, uid, did, f"lyttord{i}", _AUDIO_PENDING)
        for i in range(2):   # новые слова, которые тупик раньше запирал
            cur = await dbc.execute(
                "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
                (f"nyord{i}", json.dumps({"translate": {"no": [f"nyord{i}"], "ru": ["новое"]},
                                          "part_of_speech": "noun"}), "noun", "A1"))
            await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                              (did, cur.lastrowid))
        await dbc.commit()
    finally:
        from db.core import _release
        await _release(dbc)
    res = await build_session(uid, size=10)
    assert res["composition"]["total"] >= 1, res["composition"]
    words = {w["no"] for w in res["words"]}
    assert any(w.startswith("nyord") for w in words), words   # новые прошли, слуховые не душат


async def test_empty_session_says_listen_pending(fresh_db, bank):
    """Если отдать нечего, а слова ждут слуха — фронт получает честную причину."""
    from tests.conftest import seed_user
    from db.learning import build_session
    uid, did = await seed_user("listener")
    dbc = await _conn()
    try:
        await _seed_learning_word(dbc, uid, did, "lytteord", _AUDIO_PENDING)
        await dbc.commit()
    finally:
        from db.core import _release
        await _release(dbc)
    res = await build_session(uid, size=10)
    assert res["composition"]["total"] == 0
    assert res["composition"].get("reason") == "listen_pending"
