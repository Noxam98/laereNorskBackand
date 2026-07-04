"""Личные наборы для изучения слов («sets» = словари hidden=0).

Проверяем осмысленно, через публичные функции:
  (1) create/list: новый набор studying=0, считается count; пустое имя и дубль → error;
  (2) add_words_to_set (массово, дедуп) + sets_for_words (карта pool→[set]);
  (3) remove_word_from_set убирает членство, но НЕ трогает прогресс SRS (user_words);
  (4) studying-тоггл: вкл → слова набора входят в Учёбу, выкл → выпадают (кроме начатых);
  (5) дрилл по набору (build_session set_id=...): только слова набора, новые как карточки
      даже когда глобально WIP/ворота закрыли бы приток; без авто-добора из «Базы»;
  (6) delete набора чистит dict_words и сохраняет user_words; нельзя удалить последний словарь.
"""
import json
import pytest

from db.core import _conn, _release, _now
from db.dictionaries import (
    create_dictionary, delete_dictionary, list_user_sets, add_words_to_set,
    remove_word_from_set, sets_for_words, get_set_words, set_dictionary_studying,
)
from db.learning import _fetch_user_words, build_session, apply_result, WIP_LIMIT, NEW_PER_SESSION
from tests.conftest import seed_user, seed_word


async def _add_pool_word(no, ru="перевод", level="A1", pos="noun"):
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
        cur = await db.execute("INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
                               (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _learning_pools(user_id, set_id=None):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id, set_id)
        return {r["pool_id"] for r in rows}
    finally:
        await _release(db)


async def _user_word_exists(user_id, pool_id):
    db = await _conn()
    try:
        async with db.execute("SELECT 1 FROM user_words WHERE user_id=? AND pool_id=?", (user_id, pool_id)) as cur:
            return (await cur.fetchone()) is not None
    finally:
        await _release(db)


# ---------------- (1) create / list ----------------

@pytest.mark.asyncio
async def test_create_and_list_set(fresh_db):
    uid, _did = await seed_user()
    res = await create_dictionary(uid, "Путешествия")
    assert "error" not in res and res["id"]
    sets = await list_user_sets(uid)
    mine = next((s for s in sets if s["name"] == "Путешествия"), None)
    assert mine is not None
    assert mine["studying"] is False          # личный набор по умолчанию НЕ в ежедневной учёбе
    assert mine["count"] == 0

    assert (await create_dictionary(uid, "  "))["error"]          # пустое имя
    assert (await create_dictionary(uid, "Путешествия"))["error"]  # дубль


# ---------------- (2) add (bulk, dedup) + membership map ----------------

@pytest.mark.asyncio
async def test_add_words_bulk_dedup_and_membership(fresh_db):
    uid, _did = await seed_user()
    sid = (await create_dictionary(uid, "Еда"))["id"]
    p1 = await _add_pool_word("brød", "хлеб")
    p2 = await _add_pool_word("melk", "молоко")

    res = await add_words_to_set(uid, sid, [p1, p2, p1])  # дубль p1 в одном вызове
    assert res["added"] == 2
    again = await add_words_to_set(uid, sid, [p1])        # повторно — уже есть
    assert again["added"] == 0

    sets = await list_user_sets(uid)
    assert next(s for s in sets if s["id"] == sid)["count"] == 2

    m = await sets_for_words(uid, [p1, p2])
    assert sid in m.get(p1, []) and sid in m.get(p2, [])

    # чужой набор — отказ
    uid2, _ = await seed_user("other")
    assert (await add_words_to_set(uid2, sid, [p1]))["error"] == "Not found"


# ---------------- (3) remove keeps SRS progress ----------------

@pytest.mark.asyncio
async def test_remove_keeps_srs_progress(fresh_db):
    uid, _did = await seed_user()
    sid = (await create_dictionary(uid, "Дом"))["id"]
    p = await _add_pool_word("dør", "дверь")
    await add_words_to_set(uid, sid, [p])
    await set_dictionary_studying(uid, sid, True)
    await apply_result(uid, p, True, mode="choice", direction="no2int")  # дать прогресс
    assert await _user_word_exists(uid, p)

    await remove_word_from_set(uid, sid, p)
    assert next(s for s in await list_user_sets(uid) if s["id"] == sid)["count"] == 0
    assert await _user_word_exists(uid, p)   # прогресс SRS остался


# ---------------- (4) studying toggle wires into daily learning ----------------

@pytest.mark.asyncio
async def test_studying_toggle_feeds_daily_learning(fresh_db):
    uid, did = await seed_user()
    # уберём слова дефолтного словаря из картины — он studying=1; просто работаем с новым набором
    sid = (await create_dictionary(uid, "Слова"))["id"]
    p = await _add_pool_word("sky", "облако")
    await add_words_to_set(uid, sid, [p])

    assert p not in await _learning_pools(uid)        # set studying=0 → не в ежедневной учёбе
    await set_dictionary_studying(uid, sid, True)
    assert p in await _learning_pools(uid)            # вкл → в учёбе
    await set_dictionary_studying(uid, sid, False)
    assert p not in await _learning_pools(uid)        # выкл → выпало (не начато)


# ---------------- (5) scoped drill ----------------

@pytest.mark.asyncio
async def test_scoped_drill_only_set_words_no_autofill(fresh_db):
    uid, did = await seed_user()
    # дефолтный словарь (studying=1) с одним словом — оно НЕ должно попасть в дрилл набора
    other, _ = await seed_word(did, "annet", "другое")

    sid = (await create_dictionary(uid, "Дрилл"))["id"]   # studying=0
    setw = [await _add_pool_word(f"w{i}", f"перевод{i}") for i in range(3)]
    await add_words_to_set(uid, sid, setw)

    # дрилл по набору: только слова набора, как карточки (новые), несмотря на studying=0
    sess = await build_session(uid, size=20, lang="ru", set_id=sid)
    pids = {e["pool_id"] for e in sess["words"]}
    assert pids == set(setw)          # ровно слова набора
    assert other not in pids          # чужое слово (другой словарь) не подмешано
    assert all(e["step"] == "card" for e in sess["words"])  # новые → карточка-интро

    # глобальная сессия НЕ видит слова выключенного набора (studying=0, не начаты)
    gsess = await build_session(uid, size=20, lang="ru")
    assert not (set(setw) & {e["pool_id"] for e in gsess["words"]})


@pytest.mark.asyncio
async def test_scoped_drill_portions_new_and_bypasses_wip(fresh_db):
    """Дрилл по набору ПОРЦИОНИРУЕТ знакомство: не больше NEW_PER_SESSION новых карточек за сессию,
    а не все слова набора сразу (баг «10 карточек подряд»). При этом WIP-лимит дрилл не ограничивает:
    набор даёт новые карточки даже когда глобально слов-в-работе уже WIP_LIMIT (обычная сессия — нет)."""
    uid, did = await seed_user()
    # забиваем глобальный WIP: WIP_LIMIT слов «в работе» в дефолтном словаре
    for i in range(WIP_LIMIT):
        pid, _ = await seed_word(did, f"work{i}", f"раб{i}")
        await apply_result(uid, pid, True, mode="choice", direction="no2int")

    # набор из НЕ начатых слов, больше потолка
    sid = (await create_dictionary(uid, "Большой"))["id"]
    words = [await _add_pool_word(f"big{i}") for i in range(NEW_PER_SESSION + 5)]
    await add_words_to_set(uid, sid, words)

    # обычная сессия при полном WIP новых слов набора НЕ вводит
    g = await build_session(uid, size=50, lang="ru")
    assert not (set(words) & {e["pool_id"] for e in g["words"]})

    # дрилл по набору: WIP игнорируется, но порция новых = ровно NEW_PER_SESSION (не все слова набора)
    sess = await build_session(uid, size=50, lang="ru", set_id=sid)
    fresh = [e for e in sess["words"] if e["pool_id"] in set(words)]
    assert len(fresh) == NEW_PER_SESSION
    assert all(e["step"] == "card" for e in fresh)


@pytest.mark.asyncio
async def test_scoped_composition_shape_matches_normal_and_has_no_reason(fresh_db):
    """ЭТАП 11 (характеризация вместо рискового response_model): composition ДРИЛЛА несёт тот же
    базовый набор ключей, что и обычная сессия, и НИКОГДА не ставит reason (session_reason для
    scoped → None — на этот контракт опирается фронт-кнопка старта). Единая строгая Pydantic-модель
    для «обычной» и scoped сессии либо потребовала бы Optional везде, либо 422-ила бы прод (см.
    docs/decoupling-plan.md, критика Этапа 11) — поэтому форму фиксируем ТЕСТОМ, а не моделью."""
    uid, did = await seed_user()
    sid = (await create_dictionary(uid, "Форма"))["id"]
    words = [await _add_pool_word(f"s{i}") for i in range(3)]
    await add_words_to_set(uid, sid, words)

    BASE = {"fresh", "review", "weak", "progress", "phrases", "grammar", "phase", "total"}
    normal = (await build_session(uid, size=20, lang="ru"))["composition"]
    scoped = (await build_session(uid, size=20, lang="ru", set_id=sid))["composition"]
    assert BASE <= set(normal) and BASE <= set(scoped)      # общий базовый контракт
    assert "reason" not in scoped                            # дрилл причину пустоты не диагностирует
    # scoped не тянет лишних ключей мимо union базы+опциональных (formsLeft/reason/formsCellsLeft)
    assert set(scoped) <= BASE | {"reason", "formsLeft", "formsCellsLeft"}


@pytest.mark.asyncio
async def test_new_per_session_setting_overrides_cap(fresh_db):
    """Настройка профиля gamePrefs.newPerSession переопределяет потолок новых карточек за сессию."""
    from db.users import set_user_game_prefs
    uid, _did = await seed_user()
    await set_user_game_prefs(uid, json.dumps({"newPerSession": 3}))
    sid = (await create_dictionary(uid, "Набор"))["id"]
    words = [await _add_pool_word(f"x{i}") for i in range(8)]
    await add_words_to_set(uid, sid, words)
    sess = await build_session(uid, size=50, lang="ru", set_id=sid)
    fresh = [e for e in sess["words"] if e["step"] == "card"]
    assert len(fresh) == 3   # потолок из настройки, не дефолтные 6


# ---------------- (6) delete cleans membership, keeps SRS; not last dict ----------------

@pytest.mark.asyncio
async def test_delete_set_cleans_membership_keeps_progress(fresh_db):
    uid, _did = await seed_user()   # есть дефолтный словарь → набор не последний
    sid = (await create_dictionary(uid, "Удаляемый"))["id"]
    p = await _add_pool_word("hund", "собака")
    await add_words_to_set(uid, sid, [p])
    await set_dictionary_studying(uid, sid, True)
    await apply_result(uid, p, True, mode="choice", direction="no2int")

    res = await delete_dictionary(uid, sid)
    assert res.get("ok")
    # членство почищено
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM dict_words WHERE dict_id=?", (sid,)) as cur:
            assert (await cur.fetchone())["c"] == 0
    finally:
        await _release(db)
    assert await _user_word_exists(uid, p)   # прогресс SRS остался
    assert sid not in {s["id"] for s in await list_user_sets(uid)}
