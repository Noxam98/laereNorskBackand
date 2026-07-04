"""ЭТАП 10: tts_backfill/embed_backfill вместо complete_batch — эмбеддинг строго по ID.

Регресс-замок инцидента autofill-homograph-requeue: эмбеддинг ДОЛЖЕН лечь в ту
запись омонима, чью пару (pid) передали, а не в старшую по написанию (get_pool_id
без pos). Иначе вектор нужной записи вечно NULL → переизбрание → слив квоты.
"""
import autofill
from db.core import _conn, _release, _now
from db.pool import get_pool_by_id, get_pool_tts


async def _row(no, pos, ru, *, embedding=None):
    import json
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at, embedding) "
            "VALUES (?,?,?,?,?,?)",
            (no, json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos}),
             pos, "A1", _now(), embedding))
        pid = cur.lastrowid
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


async def test_embed_backfill_writes_to_given_id_not_senior(fresh_db, monkeypatch):
    # омонимы «bank»: старшая (сущ.) УЖЕ с вектором, младшая (глаг.) без
    senior = await _row("bank", "noun", "банк", embedding=b"OLDVEC")
    junior = await _row("bank", "verb", "стучать", embedding=None)

    monkeypatch.setattr(autofill.llm, "embed_enabled", lambda: True)

    async def fake_embed(texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(autofill, "embed_texts", fake_embed)

    n = await autofill.embed_backfill([(junior, "bank")])
    assert n == 1
    assert (await get_pool_by_id(junior))["embedding"]            # младшая получила вектор
    assert (await get_pool_by_id(senior))["embedding"] == b"OLDVEC"  # старшая НЕ тронута


async def test_embed_backfill_skips_already_embedded(fresh_db, monkeypatch):
    pid = await _row("hus", "noun", "дом", embedding=b"HAVE")
    monkeypatch.setattr(autofill.llm, "embed_enabled", lambda: True)

    async def boom(texts):
        raise AssertionError("не должны звать embed для записи с вектором")

    monkeypatch.setattr(autofill, "embed_texts", boom)
    assert await autofill.embed_backfill([(pid, "hus")]) == 0


async def test_embed_backfill_paused_noop(fresh_db, monkeypatch):
    pid = await _row("bil", "noun", "машина", embedding=None)
    monkeypatch.setattr(autofill.llm, "embed_enabled", lambda: True)
    monkeypatch.setitem(autofill.runtime.PAUSED, "embed", True)
    assert await autofill.embed_backfill([(pid, "bil")]) == 0
    monkeypatch.setitem(autofill.runtime.PAUSED, "embed", False)


async def test_tts_backfill_by_word_dedups_spelling(fresh_db, monkeypatch):
    await _row("katt", "noun", "кошка")
    calls = []

    async def fake_synth(w):
        calls.append(w)
        return b"MP3"

    monkeypatch.setattr(autofill, "synth_tts", fake_synth)
    made = await autofill.tts_backfill(["katt", "katt", "KATT"])   # дубли/регистр → один синтез
    assert made == 1 and len(calls) == 1
    assert await get_pool_tts("katt") == b"MP3"
