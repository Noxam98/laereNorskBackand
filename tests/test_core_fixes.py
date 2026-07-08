"""Регрессии по багам ядра SRS (полоса: db/learning, srs/steps, srs/status, session/pools,
db/learning_forms). Каждый тест назван по номеру фикса из задания.

Чистые функции проверяем значениями; DB-пути — через fresh_db + seed_user/seed_word.
"""
import json

import pytest

from db.core import _conn, _release, _now
from db.learning import build_session, build_listen_session, apply_result
from tests.conftest import seed_user, seed_word


async def _upsert_state(uid, pid, modes=None, forms=None, correct=0, incorrect=0):
    """Прямо подкрутить состояние слова (modes/forms/попытки) — моделируем середину рампы."""
    db = await _conn()
    try:
        await db.execute(
            "INSERT OR IGNORE INTO user_words (user_id, pool_id, created_at) VALUES (?,?,?)",
            (uid, pid, _now()))
        await db.execute(
            "UPDATE user_words SET modes = ?, correct = ?, incorrect = ?, last_seen = ? "
            "WHERE user_id = ? AND pool_id = ?",
            (json.dumps(modes or {}), correct, incorrect, "2000-01-01T00:00:00", uid, pid))
        if forms is not None:
            await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?", (forms, pid))
        await db.commit()
    finally:
        await _release(db)


async def _get_modes(uid, pid):
    db = await _conn()
    try:
        async with db.execute(
                "SELECT modes FROM user_words WHERE user_id = ? AND pool_id = ?", (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    return json.loads(r["modes"] or "{}") if r else {}


# ── #1: битая строка forms не роняет сборку сессии ────────────────────────────
async def test_broken_forms_does_not_crash_build_session(fresh_db):
    """Голый json.loads(forms) в make_element ронял ВСЮ сессию на одной битой строке.
    Пред-парсинг в _load (try/except→None) → сборка проходит."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    await _upsert_state(uid, pid, forms="{это не JSON")   # битая колонка forms
    res = await build_session(uid, size=20)               # не должно бросить
    assert isinstance(res, dict) and "words" in res
    # слово попало в сессию с forms=None (а не упало на json.loads)
    got = [w for w in res["words"] if w["pool_id"] == pid]
    assert got and got[0]["forms"] is None


async def test_broken_forms_does_not_crash_listen_session(fresh_db):
    """Та же защита в build_listen_session (отдельная выборка, не _load)."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    # текстовая рампа сдана, аудио-клетка (choice_no2int) — нет → «ждёт слух»; forms битый
    await _upsert_state(uid, pid,
                        modes={"choice_int2no": "1", "build_int2no": "1", "input_int2no": "1"},
                        forms="{битый", correct=3)
    res = await build_listen_session(uid, size=20)        # не должно бросить
    got = [w for w in res["words"] if w["pool_id"] == pid]
    assert got and got[0]["forms"] is None


# ── #2: early_review_pool фильтрует (review_step), ПОТОМ режет ─────────────────
def test_early_review_filters_then_slices():
    from session.pools import early_review_pool

    def mk(pid, due):
        return {"status": "learning", "due": False,
                "row": {"pool_id": pid, "correct": 1, "incorrect": 0,
                        "due_at": due, "archived": 0, "known": 0}}

    els = [mk(1, "2020-01-01"), mk(2, "2020-01-02"),
           mk(3, "2020-01-03"), mk(4, "2020-01-04")]
    deaf = {1, 2}   # «глухие» (слуховые у audio-юзера): review_step → None

    def review_step(e):
        pid = e["row"]["pool_id"]
        return None if pid in deaf else ("input_int2no", "input", "int2no")

    out = early_review_pool(els, review_step=review_step, size=2)
    # срез ДО фильтра дал бы пусто (первые 2 по due — глухие); фильтр ПОТОМ срез → берём 3,4
    assert [e["row"]["pool_id"] for e, _ in out] == [3, 4]
    assert len(out) == 2


# ── #5: быстрый верный ответ на НОВОМ → интервал сразу 2 (бонус был мёртвым) ──
def test_form_fast_bonus_interval_two():
    from db.learning_forms import schedule_form
    # produce, новая клетка (interval_days=0), быстрый верный → 2 (а не 1)
    ns, _ease, iv, due = schedule_form("produce", 2.5, 0, True, elapsed=1.0)
    assert ns == "produce" and iv == 2 and due == 2
    # медленный верный на новой клетке → 1 (бонус не даём)
    _, _e, iv_slow, _d = schedule_form("produce", 2.5, 0, True, elapsed=100.0)
    assert iv_slow == 1


async def test_base_fast_bonus_interval_two(fresh_db):
    """Тот же бонус в base-SRS (db/learning.apply_result): первый быстрый верный ответ на
    новом слове → interval_days = 2."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    await apply_result(uid, pid, True, elapsed=1.0, mode="choice", direction="int2no")
    db = await _conn()
    try:
        async with db.execute(
                "SELECT interval_days FROM user_words WHERE user_id = ? AND pool_id = ?",
                (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    assert r["interval_days"] == 2


# ── #6: review_step у FUNC_CHOICE не срезает choice_no2int (это ТЕКСТ, не слух) ─
def test_review_step_func_choice_keeps_no2int():
    from srs import steps
    from srs.cells import FUNC_CHOICE, CONTENT
    # FUNC_CHOICE: choice_no2int — текстовый выбор → повтор идёт по нему
    assert steps.review_step(FUNC_CHOICE, {}) == ("choice_no2int", "choice", "no2int")
    # CONTENT: аудио-клетка (choice_no2int) из повтора вырезана → последний текст input_int2no
    done = {c: "1" for c in ("choice_int2no", "choice_no2int", "build_int2no", "input_int2no")}
    assert steps.review_step(CONTENT, done) == ("input_int2no", "input", "int2no")


# ── #3: откат ступени учитывает audio_on и не путает аудио-клетку ─────────────
async def test_rollback_audio_on_skips_audio_cell(fresh_db):
    """audio ВКЛ (дефолт): ошибка на build откатывает предыдущий ТЕКСТОВЫЙ шаг (choice_int2no),
    а не аудио-клетку choice_no2int."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    await _upsert_state(uid, pid, modes={"choice_int2no": "1", "hist": "1"}, correct=1)
    await apply_result(uid, pid, False, elapsed=2.0, mode="build", direction="int2no")
    m = await _get_modes(uid, pid)
    assert m.get("build_int2no") == ""       # текущая клетка сброшена
    assert m.get("choice_int2no") == ""      # откат на ПРЕДЫДУЩИЙ ТЕКСТОВЫЙ шаг (не на аудио-клетку)


async def test_rollback_audio_off_resets_no2int(fresh_db):
    """audio ВЫКЛ: choice_no2int — обычная текстовая клетка → ошибка на build откатывает её."""
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    # выключаем аудио в профиле
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_prefs = ? WHERE id = ?",
                         (json.dumps({"audio": False}), uid))
        await db.commit()
    finally:
        await _release(db)
    await _upsert_state(uid, pid,
                        modes={"choice_int2no": "1", "choice_no2int": "1", "hist": "11"}, correct=2)
    await apply_result(uid, pid, False, elapsed=2.0, mode="build", direction="int2no")
    m = await _get_modes(uid, pid)
    assert m.get("build_int2no") == ""       # текущая клетка сброшена
    assert m.get("choice_no2int") == ""      # предыдущий шаг (при audio ВЫКЛ это текст) откачен


# ── #4: per-POS тумблер грамматики гейтит попадание слова в форм-батч ─────────
async def _master_content_word(uid, pid):
    """Довести контентное слово до mastered через apply_result (все 4 клетки рампы верно)."""
    for mode, direction in (("choice", "int2no"), ("choice", "no2int"),
                            ("build", "int2no"), ("input", "int2no")):
        await apply_result(uid, pid, True, elapsed=1.0, mode=mode, direction=direction)


_NOUN_FORMS = json.dumps({"gender": "et", "indef_pl": "hus", "def_sg": "huset", "def_pl": "husene"})


async def test_form_batch_gated_by_pos_toggle_off(fresh_db):
    """POS выключен тумблером (grammarPos.noun=false) → выученное сущ. НЕ копится в форм-батч
    (критерий совпадает с build_universe; иначе цикл churn'ит forms↔words)."""
    from db.learning_forms import get_form_cycle
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом", pos="noun")
    await _upsert_state(uid, pid, forms=_NOUN_FORMS)
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_prefs = ? WHERE id = ?",
                         (json.dumps({"grammar": True, "grammarPos": {"noun": False}}), uid))
        await db.commit()
    finally:
        await _release(db)
    await _master_content_word(uid, pid)
    cyc = await get_form_cycle(uid)
    assert cyc is None or pid not in cyc["batch"]   # выключенный POS в батч не попал


async def test_form_batch_added_when_pos_enabled(fresh_db):
    """Контроль: POS включён (дефолт) → выученное формо-способное сущ. попадает в батч."""
    from db.learning_forms import get_form_cycle
    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом", pos="noun")
    await _upsert_state(uid, pid, forms=_NOUN_FORMS)
    await _master_content_word(uid, pid)
    cyc = await get_form_cycle(uid)
    assert cyc is not None and pid in cyc["batch"]


# ── #8: играбельная фраза без достижимых дистракторов — архивируется ──────────
async def test_unsolvable_playable_phrase_archived(fresh_db):
    """Фраза (pos=phrase, ≥2 дистрактора), у которой дистракторы — свободный текст (не из Базы),
    вечно висела бы в WIP (order-ветка всегда continue). build_session её архивирует."""
    uid, did = await seed_user()
    # фраза с игровыми дистракторами, которых НЕТ в word_pool (не pool-backed)
    data = json.dumps({"translate": {"no": ["ha det bra"], "ru": ["пока"]},
                       "part_of_speech": "phrase",
                       "game": {"distractors": ["zzz aaa", "qqq www", "eee rrr"]}})
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, level, created_at) VALUES (?,?,?,?)",
            ("ha det bra", data, "A1", _now()))
        pid = cur.lastrowid
        await db.execute("INSERT INTO dict_words (dict_id, pool_id, created_at) VALUES (?,?,?)",
                         (did, pid, _now()))
        await db.commit()
    finally:
        await _release(db)
    # прошли первую клетку (choice_no2int) → следующий шаг рампы фразы = order
    await _upsert_state(uid, pid, modes={"choice_no2int": "1", "hist": "1"}, correct=1)
    await build_session(uid, size=20)
    db = await _conn()
    try:
        async with db.execute(
                "SELECT archived FROM user_words WHERE user_id = ? AND pool_id = ?",
                (uid, pid)) as cur:
            r = await cur.fetchone()
    finally:
        await _release(db)
    assert r is not None and r["archived"] == 1
