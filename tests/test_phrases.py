"""Рампа устойчивых выражений (pos='phrase'): карточка → choice_no2int → order_int2no.
Проверяем нормализацию/валидацию генерации, клетки рампы, грейдинг до mastered и что
build_session отдаёт элемент игры order с дистракторами."""
import json
import pytest

from db.core import _conn, _release, _now
from db.learning import required_cells, PHRASE_CELLS, apply_result, build_session, suggest_phrases, PHRASE_BUFFER
from llm.phrases import clean_phrase_item
from tests.conftest import seed_user


async def _seed_phrase_pool(no, distractors=("a", "b"), level="A1", game=True):
    """Фраза только в общий пул (game-ready по умолчанию), без привязки к юзеру."""
    db = await _conn()
    try:
        data = {"word": no, "part_of_speech": "phrase", "translate": {"no": [no], "ru": ["x"]}}
        if game:
            data["game"] = {"distractors": list(distractors)}
        cur = await db.execute("INSERT INTO word_pool (norwegian,data,level,pos,created_at) VALUES (?,?,?,?,?)",
                               (no, json.dumps(data, ensure_ascii=False), level, "phrase", _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _seed_word(no, level="A1", pos=""):
    """Обычное слово в пул (для дистракторов: фильтр узнаваемости требует их наличия в Базе)."""
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO word_pool (norwegian,data,level,pos,created_at) VALUES (?,?,?,?,?)",
                               (no, json.dumps({"translate": {"no": [no]}}), level, pos, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _seed_phrase(dict_id, no="ta bussen", distractors=("toget", "bilen"), level="A1", distr_level="A1", seed_distr=True):
    """Запись пула pos='phrase' с game.distractors + привязка к словарю юзера. Дистракторы тоже
    заводим в Базу с уровнем distr_level — иначе фильтр узнаваемости их отсеет (нет в Базе).
    seed_distr=False — тест сам заведёт дистракторы (нужные уровни)."""
    if seed_distr:
        for d in distractors:
            await _seed_word(d, level=distr_level)
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
    assert required_cells(row) == PHRASE_CELLS == ("choice_no2int", "order_int2no",
                                                   "build_int2no", "input_int2no")


@pytest.mark.asyncio
async def test_phrase_without_distractors_falls_back_to_word_ramp(fresh_db):
    """Гард: phrase-запись без game.distractors (легаси/неполная) НЕ уходит на order-рампу,
    а остаётся на обычной рампе — чтобы не сломать уже учащиеся слова шагом без фронт-игры."""
    from db.learning import REQUIRED_CELLS
    row = {"norwegian": "dukke opp", "data": json.dumps({"part_of_speech": "phrase",
           "translate": {"no": ["dukke opp"], "ru": ["появиться"]}})}
    assert required_cells(row) == REQUIRED_CELLS


@pytest.mark.asyncio
async def test_phrase_masters_through_four_cells(fresh_db):
    """Рампа фразы: выбор → порядок слов → СБОРКА ИЗ БУКВ → свободный ввод."""
    uid, did = await seed_user()
    pid = await _seed_phrase(did)
    await apply_result(uid, pid, True, mode="study", direction=None)          # карточка — не в зачёт
    r = await apply_result(uid, pid, True, mode="choice", direction="no2int")
    assert r["mastered"] is False
    r = await apply_result(uid, pid, True, mode="order", direction="int2no")
    assert r["mastered"] is False
    r = await apply_result(uid, pid, True, mode="build", direction="int2no")  # новая ступень
    assert r["mastered"] is False                                            # ещё клетка input впереди
    r = await apply_result(uid, pid, True, mode="input", direction="int2no")
    assert r["mastered"] is True                                             # все четыре клетки пройдены


@pytest.mark.asyncio
async def test_phrase_step_order_puts_build_before_input(fresh_db):
    """После «порядка слов» следующей выдаётся именно сборка из букв, а не сразу ввод."""
    from srs.steps import next_step
    from srs.cells import PHRASE
    modes = {"choice_no2int": "1", "order_int2no": "1"}
    assert next_step(PHRASE, modes, attempts=3, audio_on=False) == ("build_int2no", "build", "int2no")
    modes["build_int2no"] = "1"
    assert next_step(PHRASE, modes, attempts=3, audio_on=False) == ("input_int2no", "input", "int2no")


def test_phrase_legacy_ramp_stays_mastered():
    """Grandfathering: кто сдал СТАРУЮ рампу (без сборки, финал — ввод по памяти), остаётся
    выученным — новая ступень не воскрешает уже доученные фразы пачкой."""
    from srs.status import is_mastered
    from srs.cells import PHRASE
    legacy = {"choice_no2int": "1", "order_int2no": "1", "input_int2no": "1"}   # build отсутствует
    assert is_mastered(PHRASE, legacy) is True
    assert next_step_none(legacy)


def next_step_none(modes):
    from srs.steps import next_step
    from srs.cells import PHRASE
    return next_step(PHRASE, modes, attempts=5, audio_on=False) is None


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


@pytest.mark.asyncio
async def test_suggest_phrases_cumulative_level_and_gameready(fresh_db):
    """Фразы подмешиваются КУМУЛЯТИВНО (level ≤ уровня юзера) и только game-ready (есть дистракторы)."""
    uid, _ = await seed_user()
    a1 = await _seed_phrase_pool("ta bussen", ["toget", "bilen"], "A1")
    a2 = await _seed_phrase_pool("gi beskjed", ["ta", "si"], "A2")
    b1 = await _seed_phrase_pool("ta vare på", ["hente", "gi"], "B1")       # выше уровня — не должна
    nog = await _seed_phrase_pool("dukke opp", level="A1", game=False)       # без дистракторов — пропуск
    res = await suggest_phrases(uid, count=10, level="A2")
    assert res["added"] == 2                                                 # A1 + A2, без B1 и без no-game
    db = await _conn()
    try:
        async with db.execute("SELECT pool_id FROM dict_words") as cur:
            ids = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert a1 in ids and a2 in ids
    assert b1 not in ids and nog not in ids


@pytest.mark.asyncio
async def test_build_session_injects_phrase_card(fresh_db):
    """build_session сам подмешивает фразу-знакомство (карточку) из пула, даже без слов у юзера."""
    uid, _ = await seed_user()
    await _seed_phrase_pool("ta bussen", ["toget", "bilen"], "A1")
    res = await build_session(uid, size=20, lang="ru")
    phr = [e for e in res["words"] if e.get("part_of_speech") == "phrase"]
    assert phr, "фраза не подмешалась в сессию"
    assert phr[0]["step"] == "card"        # вводится карточкой-знакомством
    assert res["composition"]["phrases"] >= 1   # счётчик фраз для карточки старта


async def _at_order(uid, pid):
    """Довести слово/фразу до шага order (пройти choice_no2int)."""
    await apply_result(uid, pid, True, mode="choice", direction="no2int")


@pytest.mark.asyncio
async def test_order_distractors_filtered_by_user_level(fresh_db):
    """В order-игре показываем только дистракторы уровня ≤ юзера: A1-юзеру C1-дистрактор отсеивается."""
    uid, did = await seed_user()                       # свежий юзер → уровень A1
    await _seed_word("toget", level="A1"); await _seed_word("bilen", level="A1")
    await _seed_word("fjellet", level="C1")            # выше уровня → отсеется
    pid = await _seed_phrase(did, "ta bussen", ["toget", "bilen", "fjellet"], seed_distr=False)
    await _at_order(uid, pid)
    res = await build_session(uid, size=20, lang="ru")
    order = [e for e in res["words"] if e.get("mode") == "order" and e["pool_id"] == pid]
    assert order, "order-элемент не появился (должно остаться 2 годных дистрактора)"
    assert set(order[0]["distractors"]) == {"toget", "bilen"}   # C1 'fjellet' отфильтрован


@pytest.mark.asyncio
async def test_phrase_skipped_and_prereq_seeded_when_few_known(fresh_db):
    """Если у фразы <2 узнаваемых дистракторов — фразу пропускаем, её невыученные дистракторы
    досыпаем в учёбу (prereq-лексика)."""
    uid, did = await seed_user()                       # уровень A1
    await _seed_word("fjellet", level="C1"); await _seed_word("huset", level="C1")
    pid = await _seed_phrase(did, "ta bussen", ["fjellet", "huset"], seed_distr=False)
    await _at_order(uid, pid)
    res = await build_session(uid, size=20, lang="ru")
    order = [e for e in res["words"] if e.get("mode") == "order" and e["pool_id"] == pid]
    assert not order, "фраза не должна показываться без узнаваемых дистракторов"
    # дистракторы досыпаны в скрытый авто-словарь юзера
    db = await _conn()
    try:
        async with db.execute(
            "SELECT wp.norwegian FROM dict_words dw JOIN word_pool wp ON wp.id=dw.pool_id "
            "JOIN dictionaries d ON d.id=dw.dict_id WHERE d.user_id=?", (uid,)) as cur:
            owned = {r["norwegian"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    assert "fjellet" in owned and "huset" in owned     # prereq-лексика добавлена
