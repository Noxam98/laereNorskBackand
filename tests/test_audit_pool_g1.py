"""Аудит-фиксы полосы G1 (routers/pool.py + db/pool.py) — модерация/доступ/DoS:

  (1) POST /pool/{word}/edit — авторизация: чужое/общее слово правит только админ; обычный
      юзер — лишь СВОЁ неодобренное (approved=0). Ревью-нейросеть авторизацией не является.
  (2) replace_pool_word синхронит колонку pos с data.part_of_speech (был рассинхрон → дубли).
  (3) get_pool_meta/get_pool_by_id прячут чужое approved=0 слово (перебор pool_id не читает pending).
  (4) get_pool_candidates (источник дистракторов/синонимов) исключает approved=0.
  (5) /pool/search клампит гигантский/отрицательный limit (без клампа LIMIT -1 = весь пул).

Стиль/фикстуры — как в остальных тестах: fresh_db + прямой вызов функций (без TestClient).
"""
import json

import pytest
from fastapi import HTTPException

from db.core import _conn, _release, _now
from db.pool import (
    get_pool_by_id, get_pool_meta, get_pool_candidates, replace_pool_word,
    _invalidate_candidates,
)
import routers.pool as RP
from routers.pool import pool_edit, pool_search
from models import PoolEditBody


OWNER, OTHER = 101, 202   # id юзеров (реальные users не нужны — сверяется только created_by)


# ── helpers ──────────────────────────────────────────────────────────────────
async def _insert(no, *, ru="перевод", pos="noun", approved=1, created_by=None,
                  embedding=b"\x01\x02"):
    """Вставить слово в пул напрямую (approved/created_by/pos под тест). Вернуть pool_id."""
    data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
    dbc = await _conn()
    try:
        cur = await dbc.execute(
            "INSERT INTO word_pool (norwegian, data, pos, approved, created_by, embedding, created_at) "
            "VALUES (?,?,?,?,?,?,?)", (no, data, pos, approved, created_by, embedding, _now()))
        pid = cur.lastrowid
        await dbc.commit()
        return pid
    finally:
        await _release(dbc)


async def _pos_of(pid):
    dbc = await _conn()
    try:
        async with dbc.execute("SELECT pos, norwegian FROM word_pool WHERE id=?", (pid,)) as cur:
            return await cur.fetchone()
    finally:
        await _release(dbc)


# ── (1) авторизация правки ───────────────────────────────────────────────────
async def test_edit_shared_word_by_non_owner_forbidden(fresh_db):
    """Общее слово (approved=1) правит НЕ-админ НЕ-автор → 403 (ревью даже не запускается)."""
    await _insert("bil", ru="машина", approved=1, created_by=None)
    with pytest.raises(HTTPException) as ei:
        await pool_edit("bil", PoolEditBody(translate={"no": ["bilen"], "ru": ["авто"]}),
                        user={"id": OTHER, "username": "bob"})
    assert ei.value.status_code == 403


async def test_edit_own_unapproved_word_allowed(fresh_db, monkeypatch):
    """Автор правит СВОЁ approved=0 слово → проходит гейт, ревью применяется, слово меняется."""
    pid = await _insert("hus", ru="дом", pos="noun", approved=0, created_by=OWNER)

    async def fake_ask(system, user, schema, **kw):    # ревью одобряет, отдаёт стандартизованное слово
        return {"approved": True, "reason": "ок",
                "word": {"word": "huset", "part_of_speech": "noun", "translate": {"ru": ["дом"]}}}
    monkeypatch.setattr(RP, "ask_json", fake_ask)

    res = await pool_edit("hus", PoolEditBody(translate={"no": ["huset"], "ru": ["дом"]}),
                          user={"id": OWNER, "username": "alice"})
    assert res.get("approved") is True and res.get("ok") is True
    assert (await _pos_of(pid))["norwegian"] == "huset"   # запись реально переписана (та же pid)


# ── (2) replace_pool_word синхронит колонку pos ──────────────────────────────
async def test_replace_pool_word_sets_pos_column(fresh_db):
    pid = await _insert("gammelt", pos="noun")            # заведомо неверный/плейсхолдерный pos
    res = await replace_pool_word("gammelt", "gammel",
                                  {"part_of_speech": "adjective", "translate": {"ru": ["старый"]}})
    assert res.get("ok") is True
    row = await _pos_of(pid)
    assert row["pos"] == "adjective"                      # колонка pos = data.part_of_speech
    assert row["norwegian"] == "gammel"


# ── (3) видимость модерации: чужое approved=0 не читается по id/norwegian ─────
async def test_by_id_and_meta_hide_other_users_pending(fresh_db):
    pid = await _insert("hemmelig", ru="секрет", approved=0, created_by=OWNER)

    # get_pool_by_id: автор видит; ЧУЖОЙ пользователь (контекст запроса, user_id передан) — нет.
    # Без user_id (внутренний/фоновый вызов — эмбеддинги, автофилл) фильтра нет: граница безопасности —
    # HTTP-роут, а он всегда передаёт аутентифицированный user_id (роуты G1 это делают).
    assert (await get_pool_by_id(pid, user_id=OWNER)) is not None
    assert (await get_pool_by_id(pid, user_id=OTHER)) is None
    assert (await get_pool_by_id(pid)) is not None            # внутренний вызов — не фильтруем

    # get_pool_meta: то же (перебор по слову чужим юзером не выдаёт чужое pending)
    assert (await get_pool_meta("hemmelig", user_id=OWNER)) is not None
    assert (await get_pool_meta("hemmelig", user_id=OTHER)) is None
    assert (await get_pool_meta("hemmelig")) is not None      # внутренний вызов — не фильтруем


async def test_by_id_approved_word_visible_to_all(fresh_db):
    pid = await _insert("apen", ru="открытый", approved=1, created_by=OWNER)
    assert (await get_pool_by_id(pid, user_id=OTHER)) is not None   # общая база видна всем
    assert (await get_pool_by_id(pid)) is not None


# ── (4) кандидаты дистракторов/синонимов исключают approved=0 ─────────────────
async def test_get_pool_candidates_excludes_unapproved(fresh_db):
    ok_pid = await _insert("godkjent", approved=1)
    bad_pid = await _insert("venter", approved=0, created_by=OWNER)
    _invalidate_candidates()                              # сбросить модульный TTL-кеш от прошлых тестов
    cands = await get_pool_candidates()
    ids = {c["id"] for c in cands}
    assert ok_pid in ids                                 # одобренное — кандидат
    assert bad_pid not in ids                            # approved=0 НЕ попадает в дистракторы/синонимы


# ── (5) /pool/search клампит limit ───────────────────────────────────────────
async def test_pool_search_clamps_limit(fresh_db):
    for i in range(55):                                  # больше кап-50 совпадений по префиксу
        await _insert(f"testord{i}", ru=f"слово{i}")
    big = await pool_search("testord", limit=100000, user={"id": OWNER})
    assert 1 <= len(big["results"]) <= 50                # без клампа вернулись бы все 55

    neg = await pool_search("testord", limit=-1, user={"id": OWNER})
    assert isinstance(neg["results"], list)
    assert 1 <= len(neg["results"]) <= 50                # без клампа LIMIT -1 = без ограничения
