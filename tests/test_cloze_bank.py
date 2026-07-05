"""Cloze служебных слов из СТАТИЧЕСКОГО банка (db.cloze_bank), а не пер-юзерной LLM-генерации.

Проверяем:
  - банк: has/items_for; поиск по (norwegian, pos), омонимы (for-предлог/for-союз) — разные items;
  - get_cloze_map строит options (перемешанные answer+distractors, «выбор из 4») и добивает
    список до 3 (клетки cloze_1..3), если валидация оставила меньше предложений;
  - рампа: служебное получает FUNC_CLOZE ТОЛЬКО если оно в банке и CLOZE_ENABLED; иначе FUNC_CHOICE;
  - level-гейт: при включённом cloze новое служебное ВНЕ банка (уровень >A2) как новое не вводится.

Банк здесь подменяется фикстурой (fake_bank) — тесты не зависят от содержимого data/cloze-bank.json."""
import json
import pytest

from db import cloze_bank
from db.core import _conn, _release, _now
from db.learning import (build_session, apply_result, get_cloze_map, ramp_kind_of,
                         required_cells)
from srs.cells import FUNC_CLOZE, FUNC_CHOICE, FUNC_CELLS, FUNC_CELLS_CHOICE
from tests.conftest import seed_user, seed_word


@pytest.fixture
def fake_bank(monkeypatch):
    """Подменить банк фикстурой (без чтения файла). 'men' — 3 предложения; 'for' — омоним
    (предлог/союз) с 1 предложением каждый (проверка добивки до 3 и разбора по pos)."""
    by_no = {
        "men": [{"pos": "conjunction", "lvl": "A1", "items": [
            {"blank": "Jeg liker kaffe, ___ ikke te.", "answer": "men", "distractors": ["eller", "for", "at"]},
            {"blank": "Huset er lite, ___ fint.", "answer": "men", "distractors": ["at", "for", "om"]},
            {"blank": "Han er gammel, ___ han jobber.", "answer": "men", "distractors": ["at", "som", "om"]},
        ]}],
        "for": [
            {"pos": "preposition", "lvl": "A1", "items": [
                {"blank": "Dette er ___ deg.", "answer": "for", "distractors": ["av", "om", "til"]}]},
            {"pos": "conjunction", "lvl": "A1", "items": [
                {"blank": "Jeg er hjemme, ___ det regner.", "answer": "for", "distractors": ["at", "som", "å"]}]},
        ],
    }
    monkeypatch.setattr(cloze_bank, "_BY_NO", by_no)
    monkeypatch.setattr(cloze_bank, "_LOADED", True)
    return by_no


# ── банк: поиск ──────────────────────────────────────────────────────────────
def test_bank_has_and_items(fake_bank):
    assert cloze_bank.has("men", "conjunction") is True
    assert cloze_bank.has("men", None) is True          # единственная запись — pos не обязателен
    assert cloze_bank.has("MEN", "conjunction") is True  # регистронезависимо
    assert cloze_bank.has("nope", "adverb") is False
    items = cloze_bank.items_for("men", "conjunction")
    assert len(items) == 3 and items[0]["answer"] == "men"


def test_bank_homograph_disambiguation(fake_bank):
    """Омоним: for-предлог и for-союз — РАЗНЫЕ предложения, выбор по pos."""
    prep = cloze_bank.items_for("for", "preposition")
    conj = cloze_bank.items_for("for", "conjunction")
    assert "deg" in prep[0]["blank"] and "regner" in conj[0]["blank"]
    assert prep[0]["blank"] != conj[0]["blank"]


# ── get_cloze_map: сборка options + добивка до 3 ─────────────────────────────
async def _seed_pool(no, pos, level="A1"):
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": ["x"]}, "part_of_speech": pos})
        cur = await db.execute("INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
                               (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_get_cloze_map_builds_options(fresh_db, fake_bank):
    pid = await _seed_pool("men", "conjunction")
    db = await _conn()
    try:
        cmap = await get_cloze_map(db, 1, [pid])
    finally:
        await _release(db)
    items = cmap[pid]
    assert len(items) == 3
    for it in items:
        assert "___" in it["blank"] and it["answer"] == "men"
        assert "men" in it["options"] and len(it["options"]) == 4   # answer + 3 дистрактора, перемешаны


@pytest.mark.asyncio
async def test_get_cloze_map_pads_to_three(fresh_db, fake_bank):
    """Банк отдал 1 предложение (for-предлог) → добиваем до 3 клеток цикличным повтором."""
    pid = await _seed_pool("for", "preposition")
    db = await _conn()
    try:
        cmap = await get_cloze_map(db, 1, [pid])
    finally:
        await _release(db)
    assert len(cmap[pid]) == 3 and all(it["answer"] == "for" for it in cmap[pid])


@pytest.mark.asyncio
async def test_get_cloze_map_surfaces_translations(fresh_db, monkeypatch):
    """Если в банке есть переводы (tr/answer_tr/distractor_tr) — get_cloze_map доносит optionsTr
    (значение каждого варианта) и sentTr (перевод предложения) для показа на разборе."""
    monkeypatch.setattr(cloze_bank, "_BY_NO", {"på": [{"pos": "preposition", "lvl": "A1", "items": [
        {"blank": "Boka ligger ___ bordet.", "answer": "på", "distractors": ["av", "om", "til"],
         "tr": {"ru": "Книга лежит на столе.", "en": "The book is on the table."},
         "answer_tr": {"ru": "на", "en": "on"},
         "distractor_tr": [{"ru": "из/от"}, {"ru": "о/об"}, {"ru": "к/до"}]}]}]})
    monkeypatch.setattr(cloze_bank, "_LOADED", True)
    pid = await _seed_pool("på", "preposition")
    db = await _conn()
    try:
        cmap = await get_cloze_map(db, 1, [pid])
    finally:
        await _release(db)
    it = cmap[pid][0]
    assert set(it["options"]) == {"på", "av", "om", "til"}       # варианты не тронуты
    assert it["sentTr"]["ru"].startswith("Книга") and it["sentTr"]["en"].endswith("table.")
    assert it["optionsTr"]["på"]["ru"] == "на"                    # ответ — контекстно
    assert it["optionsTr"]["av"]["ru"] == "из/от"                 # дистрактор — общее значение


@pytest.mark.asyncio
async def test_get_cloze_map_skips_unbanked(fresh_db, fake_bank):
    """Служебное вне банка → в карте его нет (на cloze-рампе оно и не окажется)."""
    pid = await _seed_pool("hos", "preposition")
    db = await _conn()
    try:
        cmap = await get_cloze_map(db, 1, [pid])
    finally:
        await _release(db)
    assert pid not in cmap


# ── рампа: FUNC_CLOZE только для банковых + kill-switch ───────────────────────
def _row(no, pos):
    return {"norwegian": no, "data": json.dumps({"part_of_speech": pos})}


def test_ramp_func_cloze_only_when_in_bank(monkeypatch, fake_bank):
    monkeypatch.setattr("db.learning.CLOZE_ENABLED", True)
    assert ramp_kind_of(_row("men", "conjunction")) == FUNC_CLOZE
    assert required_cells(_row("men", "conjunction")) == FUNC_CELLS
    # служебное, но вне банка (уровень >A2 / не покрыто) → упрощённая рампа «только выбор»
    assert ramp_kind_of(_row("hos", "preposition")) == FUNC_CHOICE
    assert required_cells(_row("hos", "preposition")) == FUNC_CELLS_CHOICE


def test_ramp_kill_switch_forces_choice(monkeypatch, fake_bank):
    """CLOZE_ENABLED=0 (общий рубильник) → даже банковое служебное идёт FUNC_CHOICE."""
    monkeypatch.setattr("db.learning.CLOZE_ENABLED", False)
    assert ramp_kind_of(_row("men", "conjunction")) == FUNC_CHOICE


# ── level-гейт: новое служебное ВНЕ банка не вводится при включённом cloze ────
async def _master(uid, pid):
    for mode, direction in (("choice", "no2int"), ("choice", "int2no"),
                            ("build", "int2no"), ("input", "int2no")):
        await apply_result(uid, pid, True, mode=mode, direction=direction)


@pytest.mark.asyncio
async def test_level_gate_blocks_unbanked_func_word(fresh_db, fake_bank, monkeypatch):
    """og нет в банке. Контентная база достаточна (content_known ≥ порога og). При выключенном
    cloze og вводится как новое (FUNC_CHOICE), при включённом — level-гейт его запирает."""
    uid, did = await seed_user()
    # 4 выученных контентных слова → content_known=4 (mastered в in_work не идут; порог og = 4)
    for i in range(4):
        pid, _ = await seed_word(did, f"noun{i}", f"сущ{i}")
        await _master(uid, pid)
    # новое служебное «og» в словаре юзера (не в банке fake_bank)
    og_pid, _ = await seed_word(did, "og", "и", pos="conjunction")

    def _served(res):
        return og_pid in [w["pool_id"] for w in res["words"]]

    monkeypatch.setattr("db.learning.CLOZE_ENABLED", False)
    assert _served(await build_session(uid, size=10)) is True     # cloze выкл → og вводится

    monkeypatch.setattr("db.learning.CLOZE_ENABLED", True)
    assert _served(await build_session(uid, size=10)) is False    # cloze вкл, og вне банка → заперт
