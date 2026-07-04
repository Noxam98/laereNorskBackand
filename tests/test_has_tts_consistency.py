"""ЭТАП 0: has_tts должен быть ОДИНАКОВЫМ через все поверхности чтения.

Инцидент #1 (3.07): вынос tts-колонки в word_tts правился в 10 местах по отдельности.
Этот тест ловит рассинхрон: одно слово с озвучкой → флаг True везде, где фронт его видит.
После Этапа 1 все места обязаны использовать единый has_tts_expr()."""
import json

from db.core import _conn, _release
from tests.conftest import seed_user

import db as D
from db.pool_queues import set_pool_tts
from db.learning import _fetch_user_words
from db.dictionaries import get_user_data


async def _seed(uid, did, no="katt"):
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, level, created_at) VALUES (?,?,?,?,datetime('now'))",
            (no, json.dumps({"translate": {"no": [no], "ru": ["кот"]},
                             "part_of_speech": "noun"}), "noun", "A1"))
        pid = cur.lastrowid
        await dbc.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,datetime('now'))",
                          (did, pid))
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


async def test_has_tts_same_everywhere(fresh_db):
    uid, did = await seed_user("tts")
    pid = await _seed(uid, did)

    async def flags():
        out = {}
        lst = await D.get_pool_list(limit=10, q="katt")
        out["pool_list"] = bool(lst["words"][0].get("hasTts") or lst["words"][0].get("has_tts"))
        meta = await D.get_pool_meta("katt")
        out["pool_meta"] = bool((meta or {}).get("hasTts") or (meta or {}).get("has_tts"))
        dbc = await _conn()
        try:
            rows = await _fetch_user_words(dbc, uid)
        finally:
            await _release(dbc)
        out["learning_rows"] = bool(rows[0]["has_tts"])
        data = await get_user_data(uid)
        w = next(w for d in data["dictList"] for w in d["words"])
        out["user_data"] = bool(w.get("hasTts") or w.get("has_tts"))
        stats = await D.get_pool_stats()
        out["admin_stats_n"] = stats["tts"]
        return out

    before = await flags()
    assert before["pool_list"] is False
    assert before["pool_meta"] is False
    assert before["learning_rows"] is False
    assert before["user_data"] is False
    n0 = before["admin_stats_n"]

    await set_pool_tts("katt", b"\x00\x01mp3")

    after = await flags()
    # ГЛАВНОЕ: все поверхности видят озвучку ОДНОВРЕМЕННО
    assert after["pool_list"] is True
    assert after["pool_meta"] is True
    assert after["learning_rows"] is True
    assert after["user_data"] is True
    assert after["admin_stats_n"] == n0 + 1
