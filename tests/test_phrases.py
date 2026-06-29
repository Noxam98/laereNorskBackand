"""Рампа устойчивых выражений (pos='phrase'): карточка → choice_no2int → order_int2no.
Проверяем нормализацию/валидацию генерации, клетки рампы, грейдинг до mastered и что
build_session отдаёт элемент игры order с дистракторами."""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning import required_cells, PHRASE_CELLS, apply_result, build_session
from llm.phrases import clean_phrase_item
from tests.conftest import seed_user


async def _seed_phrase(dict_id, no="ta bussen", distractors=("toget", "bilen"), level="A1"):
    """Запись пула pos='phrase' с game.distractors + привязка к словарю юзера."""
    db = await _conn()
    try:
        data = json.dumps({
            "word": no, "part_of_speech": "phrase",
            "translate": {"no": [no], "ru": ["сесть на автобус"]},
            "example": "jeg må ta bussen", "subtype": "collocation",
            "game": {"distractors": list(distractors)},
        }, ensure_ascii=False)
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,pos,created_at) VALUES (?,?,?,?,?)",
            (no, data, level, "phrase", _now()))
        pid = cur.lastrowid
        await db.execute("INSERT INTO dict_words (dict_id,pool_id,created_at) VALUES (?,?,?)", (dict_id, pid, _now()))
        await db.commit()
        return pid
    finally:
        await _release(db)


def test_clean_phrase_item_normalizes_and_validates():
    # снимает 'å'/точку/регистр; откидывает мультисловный дистрактор и дистрактор из самой фразы
    it = clean_phrase_item({
        "phrase": "Å Ta Bussen.", "subtype": "collocation", "level": "A1",
        "translate": {"ru": ["сесть на автобус"], "en": ["take the bus"]},
        "example": "jeg må ta bussen",
        "distractors": ["toget", "et tog", "bussen", "bilen"],   # 'et tog' мультислово, 'bussen' — из фразы
    })
    assert it["word"] == "ta bussen"
    assert it["part_of_speech"] == "phrase"
    assert it["game"]["distractors"] == ["toget", "bilen"]
    assert it["level"] == "A1" and it["subtype"] == "collocation"


def test_clean_phrase_item_rejects_bad():
    assert clean_phrase_item({"phrase": "hei", "distractors": ["a", "b"], "translate": {"ru": ["привет"]}}) is None  # 1 слово
    assert clean_phrase_item({"phrase": "ta bussen", "distractors": ["toget"], "translate": {"ru": ["x"]}}) is None  # <2 дистр
    assert clean_phrase_item({"phrase": "ta bussen", "distractors": ["a", "b"], "translate": {}}) is None            # нет перевода


@pytest.mark.asyncio
async def test_phrase_ramp_cells(fresh_db):
    uid, did = await seed_user()
    pid = await _seed_phrase(did)
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian, data FROM word_pool WHERE id=?", (pid,)) as cur:
            row = dict(await cur.fetchone())
    finally:
        await _release(db)
    assert required_cells(row) == PHRASE_CELLS == ["choice_no2int", "order_int2no"]


@pytest.mark.asyncio
async def test_phrase_masters_through_two_cells(fresh_db):
    uid, did = await seed_user()
    pid = await _seed_phrase(did)
    await apply_result(uid, pid, True, mode="study", direction=None)          # карточка — не в зачёт
    r = await apply_result(uid, pid, True, mode="choice", direction="no2int")
    assert r["mastered"] is False
    r = await apply_result(uid, pid, True, mode="order", direction="int2no")
    assert r["mastered"] is True                                              # обе клетки рампы пройдены


@pytest.mark.asyncio
async def test_build_session_emits_order_with_distractors(fresh_db):
    uid, did = await seed_user()
    pid = await _seed_phrase(did, distractors=("toget", "bilen", "sykkelen"))
    await apply_result(uid, pid, True, mode="choice", direction="no2int")     # прошли 1-ю клетку → дальше order
    res = await build_session(uid, size=20, lang="ru")
    order = [e for e in res["words"] if e.get("mode") == "order" and e["pool_id"] == pid]
    assert order, "order-элемент не появился в сессии"
    el = order[0]
    assert el["direction"] == "int2no" and el["step"] == "order_int2no"
    assert el["distractors"] == ["toget", "bilen", "sykkelen"]
