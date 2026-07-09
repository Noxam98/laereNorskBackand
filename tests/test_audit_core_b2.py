"""Регрессии по аудиту ядра «Учёбы» (полоса B2): db/learning, db/learning_suggest.

Проверяем фиксы:
  #5  estimate_level игнорирует «голый» archived (archived=1, mastered=0 не поднимает уровень);
  #6  due_count в learning_stats не считает выученные СЛУЖЕБНЫЕ слова (у них нет текстового повтора);
  #7  _activity_metrics ключуется по UTC-дате (_today), а не по локальной date.today();
  #3  overlay-путь apply_result создаёт пропавшую строку user_words (INSERT OR IGNORE) перед UPDATE;
  #8  гистограмма by_level.done согласована с currentLevel на полосе 250–499 (не «A1 done» + «на A1»);
  #9  слуховая партия становится ready в «хвосте», когда приток новых иссяк, а pending < pack.

Чистые пути — значениями; DB-пути — через fresh_db + seed_user/seed_word (conftest).
"""
import json
from datetime import date

import db.learning as learning
from db.core import _conn, _release, _now
from db.learning import (
    apply_result, listen_status, learning_stats, estimate_level, _activity_metrics,
)
from tests.conftest import seed_user, seed_word

_CONTENT_MASTERED = {"choice_int2no": "1", "choice_no2int": "1",
                     "build_int2no": "1", "input_int2no": "1"}   # все клетки CONTENT-рампы пройдены


async def _set_word_state(uid, pid, **cols):
    """INSERT OR IGNORE строку user_words + проставить произвольные колонки (modes как JSON)."""
    if "modes" in cols and not isinstance(cols["modes"], str):
        cols["modes"] = json.dumps(cols["modes"] or {})
    db = await _conn()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO user_words (user_id, pool_id, created_at) VALUES (?,?,?)",
            (uid, pid, _now()))
        sets = ", ".join(f"{k} = ?" for k in cols)
        await db.execute(f"UPDATE user_words SET {sets} WHERE user_id = ? AND pool_id = ?",
                         (*cols.values(), uid, pid))
        await db.commit()
    finally:
        await _release(db)


async def _bulk_words(uid, did, n, *, prefix, mastered=0, archived=0, correct=0,
                      modes=None, level="A1", pos="noun", link_dict=True):
    """Массово создать n слов пула (+ dict_words при link_dict) и их состояние user_words.
    Дёшево (executemany) — для тестов, где нужно перевалить кумулятивный CEFR-порог. Возвращает pids."""
    now = _now()
    data = json.dumps({"translate": {"no": ["w"], "ru": ["п"]}, "part_of_speech": pos})
    modes_json = json.dumps(modes or {})
    db = await _conn()
    try:
        await db.executemany(
            "INSERT INTO word_pool (norwegian, data, level, created_at) VALUES (?,?,?,?)",
            [(f"{prefix}{i}", data, level, now) for i in range(n)])
        async with db.execute(
                "SELECT id FROM word_pool WHERE norwegian LIKE ? ORDER BY id", (f"{prefix}%",)) as cur:
            pids = [r["id"] for r in await cur.fetchall()]
        if link_dict:
            await db.executemany(
                "INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,?)",
                [(did, pid, now) for pid in pids])
        await db.executemany(
            "INSERT INTO user_words (user_id, pool_id, modes, correct, mastered, archived, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            [(uid, pid, modes_json, correct, mastered, archived, now) for pid in pids])
        await db.commit()
        return pids
    finally:
        await _release(db)


# ── #5: estimate_level игнорирует «голый» archived (мягкое удаление ≠ выучено) ──
async def test_estimate_level_ignores_bare_archived(fresh_db):
    """learning_remove ставит archived=1, mastered=0 — это НЕ «выучено». 500 таких строк не должны
    поднять уровень (иначе estimate_level расходится с learning_stats.currentLevel). Контроль: те же
    строки как mastered=1 → уровень поднимается (порог A2 = 500)."""
    uid, did = await seed_user()
    await _bulk_words(uid, did, 500, prefix="arch", archived=1, mastered=0, link_dict=False)
    assert await estimate_level(uid) == "A1"        # 500 «голых» archived не считаются выученными
    db = await _conn()
    try:
        await db.execute("UPDATE user_words SET mastered = 1, archived = 0 WHERE user_id = ?", (uid,))
        await db.commit()
    finally:
        await _release(db)
    assert await estimate_level(uid) == "A2"        # те же 500, но mastered → уровень поднялся


# ── #6: due_count не считает выученные СЛУЖЕБНЫЕ слова (нет текстового повтора) ─
async def test_due_count_excludes_mastered_function_words(fresh_db):
    """Выученное служебное слово с наступившим due_at build_session текстовым повтором НЕ обслуживает
    (у служебных cloze-рампа) — его вестигиальный due не должен надувать «к повторению». Контентное
    выученное с due считается (due_mastered). Ожидаем due == 1 (только контентное)."""
    uid, did = await seed_user()
    fpid, _ = await seed_word(did, "til", "к", pos="preposition")   # служебное (FUNCTION_POS)
    cpid, _ = await seed_word(did, "hus", "дом", pos="noun")        # контентное
    past = "2000-01-01T00:00:00"
    # служебное: FUNC_CHOICE-рампа сдана (обе choice-клетки), mastered, due наступил, не сертиф.
    await _set_word_state(uid, fpid, modes={"choice_int2no": "1", "choice_no2int": "1"},
                          correct=2, mastered=1, due_at=past, certified=0, archived=0, known=0)
    # контентное: вся CONTENT-рампа сдана, mastered, due наступил, не сертиф.
    await _set_word_state(uid, cpid, modes=_CONTENT_MASTERED,
                          correct=4, mastered=1, due_at=past, certified=0, archived=0, known=0)
    stats = await learning_stats(uid)
    assert stats["due"] == 1        # служебное исключено, контентное посчитано


# ── #7: _activity_metrics читает журнал по UTC-дате (_today), не date.today() ──
async def test_activity_metrics_keys_off_utc(fresh_db, monkeypatch):
    """Строки user_activity пишутся ключом _now()[:10] (UTC). _activity_metrics должен читать той же
    UTC-датой (_today). Подменяем _today на фикс-дату и проверяем, что журнал этой даты попал в 'today'.
    При старом date.today() метрика искала бы ЛОКАЛЬНУЮ дату → сегодняшних данных не нашла бы (done=0)."""
    uid, _ = await seed_user()
    fixed = date(2030, 1, 15)
    monkeypatch.setattr(learning, "_today", lambda: fixed)
    db = await _conn()
    try:
        await db.execute(
            "INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,?,?)",
            (uid, fixed.isoformat(), 5, 3))
        await db.commit()
        m = await _activity_metrics(db, uid)
    finally:
        await _release(db)
    assert m["today"]["done"] == 5      # журнал UTC-даты прочитан как «сегодня»
    assert m["streak"] >= 1             # активный день = _today() → стрик ≥ 1
    assert m["accuracy"] == 60          # round(100*3/5)


# ── #3: overlay-путь apply_result создаёт пропавшую строку user_words ──────────
async def test_overlay_branch_creates_missing_row(fresh_db):
    """Грамм-overlay (★) отвечал только UPDATE — если строка user_words исчезла между сборкой сессии и
    ответом (learning_remove и т.п.), прогресс формы уходил «в пустоту». Теперь INSERT OR IGNORE создаёт
    строку. Отвечаем по choice_gender у сущ. с родом, НЕ создавая строку заранее — она должна появиться."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом", pos="noun")
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?",
                         (json.dumps({"gender": "en"}), pid))
        await db.commit()
        # строки user_words для (uid, pid) СПЕЦИАЛЬНО ещё нет
        async with db.execute(
                "SELECT COUNT(*) c FROM user_words WHERE user_id = ? AND pool_id = ?", (uid, pid)) as cur:
            assert (await cur.fetchone())["c"] == 0
    finally:
        await _release(db)
    # overlay-клетка choice_gender (mode=choice, direction=gender): в gcells, НЕ в base-рампе → overlay-ветка
    await apply_result(uid, pid, True, elapsed=1.0, mode="choice", direction="gender")
    db = await _conn()
    try:
        async with db.execute(
                "SELECT modes FROM user_words WHERE user_id = ? AND pool_id = ?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    assert r is not None                                   # строка создана (была бы None до фикса)
    assert json.loads(r["modes"] or "{}").get("choice_gender") == "1"   # прогресс формы записан


# ── #8: by_level.done согласован с currentLevel на полосе 250–499 ──────────────
async def test_histogram_done_matches_current_level_mid_band(fresh_db):
    """На кумулятиве 250–499 юзер НАходится на A1 (не прошёл его). done[A1] должен быть False — иначе UI
    показывал бы «A1 done» и одновременно «на A1 → A2» (target как «пройти lv» против «достичь lv»)."""
    uid, did = await seed_user()
    await _bulk_words(uid, did, 300, prefix="cw", mastered=1, correct=4, modes=_CONTENT_MASTERED)
    stats = await learning_stats(uid)
    assert stats["currentLevel"] == "A1"                   # 300 выученных → всё ещё A1 (порог A2 = 500)
    assert stats["byLevel"]["A1"]["done"] is False         # ты НА A1, не прошёл его (было True до фикса)
    assert stats["byLevel"]["A2"]["done"] is False


# ── #9: слуховая партия ready в «хвосте», когда приток новых иссяк ─────────────
async def test_listen_ready_at_exhausted_tail(fresh_db):
    """Единственное слово ждёт слух, а расти пачке больше не из чего (нет новых/учащихся контентных):
    ready должен стать True при pending>0, иначе слово зависло бы «невыучиваемым» (pending < pack навсегда)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом", pos="noun")
    # текстовая рампа сдана, аудио-клетка (choice_no2int) — нет → «ждёт слух»
    await _set_word_state(uid, pid, correct=3,
                          modes={"choice_int2no": "1", "build_int2no": "1", "input_int2no": "1"})
    st = await listen_status(uid)
    assert st["audio"] is True and st["pending"] == 1
    assert st["pending"] < st["pack"]                      # порог не набран
    assert st["ready"] is True                             # но приток иссяк → хвост открыт


async def test_listen_not_ready_when_new_words_remain(fresh_db):
    """Контроль анти-преждевременности: если есть новое контентное слово (пачка ещё может дорасти),
    ready НЕ включается на неполном pending (growable > 0 → порог прежний)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом", pos="noun")
    await _set_word_state(uid, pid, correct=3,
                          modes={"choice_int2no": "1", "build_int2no": "1", "input_int2no": "1"})
    await seed_word(did, "bil", "машина", pos="noun")      # новое контентное — пачке ещё есть куда расти
    st = await listen_status(uid)
    assert st["pending"] == 1 and st["pending"] < st["pack"]
    assert st["ready"] is False                            # рано слух не открываем
