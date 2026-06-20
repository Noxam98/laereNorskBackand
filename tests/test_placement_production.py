"""Входной тест: ПРОДУКЦИЯ на верхних уровнях.

Нижние уровни (A1,A2,B1) — «выбор из 4» (узнавание), верхние (B2,C1,C2) — «ввод»
(продукция норвежского). Узнавание угадывается → завышает уровень; ввод угадать нельзя,
поэтому на верхах честнее.

Единый контракт вопроса (важно — фронт делается параллельно под него):
- choice (A1,A2,B1): {no:<норв. слово>, level, type:"choice", options:[4 перевода на lang]}
- input  (B2,C1,C2): {no:<норв. лемма — КЛЮЧ для грейда>, prompt:<перевод на lang>,
  level, type:"input"} (без options)

Клиент присылает: [{no, level, answer, type}].
grade_placement:
- choice: верно если answer ∈ translations[lang] слова no;
- input:  верно если введённое совпало с НОРВЕЖСКОЙ леммой no (или любым translate.no),
  СНИСХОДИТЕЛЬНО (lower/trim, å→a ø→o æ→ae, срез прочих диакритик — как foldLoose на фронте).
Калибровку НЕ ослабляем: per=8, порог 0.8.

Тесты проверяют поведение контракта, а не подгоняются под реализацию.
"""
import json
import pytest

from db.learning import (
    build_placement, grade_placement, get_start_level,
    LEVELS, PLACEMENT_INPUT_LEVELS,
)
from db.core import _conn, _release, _now
from tests.conftest import seed_user


async def _seed_pool_word(no, ru, level="A1", pos="noun"):
    """Положить слово прямо в пул (placement читает word_pool, словарь не нужен)."""
    db = await _conn()
    try:
        data = json.dumps({"translate": {"no": [no], "ru": [ru]}, "part_of_speech": pos})
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
            (no, data, level, _now()))
        await db.commit()
        return cur.lastrowid
    finally:
        await _release(db)


async def _seed_full_pool(per_level=12, pos=("noun",)):
    """Засеять пул, чтобы на каждом уровне хватало слов под вопросы и дистракторы.
    Возвращает {level: {ru: no}} для разбора ответов. Норвежское — нижним регистром:
    get_pool_id ищет по normalize_word (strip+lower) с точным сравнением."""
    by_level = {}
    for lv in LEVELS:
        by_level[lv] = {}
        for p in pos:
            for i in range(per_level):
                no = f"{lv}_{p}_{i}_no".lower()
                ru = f"{lv}_{p}_{i}_ru"
                await _seed_pool_word(no, ru, level=lv, pos=p)
                by_level[lv][ru] = no
    return by_level


# (1) build_placement: A1/A2/B1 — choice с options; B2/C1/C2 — input с prompt(перевод),
#     без options; no — норвежская лемма. Перевод-prompt совпадает с переводом леммы на lang.
async def test_build_question_contract_choice_vs_input(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos=("noun",))
    correct_ru = {no: ru for lv in LEVELS for ru, no in by_level[lv].items()}
    no_by_level = {lv: set(by_level[lv].values()) for lv in LEVELS}

    res = await build_placement(lang="ru", per=8)
    qs = res["questions"]
    assert qs, "входной тест должен сгенерировать вопросы"

    # оба типа присутствуют
    assert any(q["type"] == "input" for q in qs), "на верхних уровнях должны быть вопросы-ввод"
    assert any(q["type"] == "choice" for q in qs), "на нижних уровнях должны быть вопросы-выбор"

    seen_input_levels, seen_choice_levels = set(), set()
    for q in qs:
        lv = q["level"]
        # no — норвежская лемма этого уровня
        assert q["no"] in no_by_level[lv], f"no {q['no']!r} не норв. лемма уровня {lv}"
        if lv in PLACEMENT_INPUT_LEVELS:
            seen_input_levels.add(lv)
            assert q["type"] == "input", f"уровень {lv} должен быть input"
            assert "options" not in q, "у input не должно быть options"
            # prompt — перевод леммы на lang (показывается пользователю)
            assert q["prompt"] == correct_ru[q["no"]], "prompt — перевод леммы на lang"
        else:
            seen_choice_levels.add(lv)
            assert q["type"] == "choice", f"уровень {lv} должен быть choice"
            assert "prompt" not in q, "у choice нет prompt — показывается само норв. слово"
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4
            assert correct_ru[q["no"]] in q["options"], "верный перевод среди вариантов"

    assert seen_input_levels == PLACEMENT_INPUT_LEVELS, "все верхние уровни — input"
    assert seen_choice_levels == set(LEVELS) - PLACEMENT_INPUT_LEVELS, "все нижние — choice"


# (2) grade input: верный норвежский ответ засчитывается; снисходительность работает
#     (Å/å, ø→o, лишние пробелы/регистр). Перевод на input-уровне НЕ засчитывается.
async def test_grade_input_lenient(fresh_db):
    uid, _ = await seed_user()
    no = "spørsmål"   # норвежская лемма с диакритиками (ø, å)
    ru = "вопрос"
    await _seed_pool_word(no, ru, level="B2", pos="noun")

    cases = [
        ("spørsmål", True),    # точное совпадение
        ("  spørsmål  ", True),  # лишние пробелы
        ("Spørsmål", True),    # регистр
        ("SPØRSMÅL", True),    # верхний регистр + диакритики
        ("sporsmal", True),    # ø→o, å→a (foldLoose)
        ("  SpOrSmAl ", True),  # регистр + пробелы + å→a/ø→o
        (ru, False),           # перевод на input-уровне не засчитывается
        ("", False),           # пусто
        ("__мимо__", False),
    ]
    for answer, expect_ok in cases:
        res = await grade_placement(
            uid, "ru", [{"no": no, "level": "B2", "answer": answer, "type": "input"}])
        got_ok = res["perLevel"]["B2"]["ok"] == 1
        assert got_ok == expect_ok, (
            f"input ответ {answer!r}: ожидали ok={expect_ok}, получили ok={got_ok}")


# (2b) input засчитывает совпадение с ЛЮБЫМ вариантом translate.no (не только с леммой no),
#      тоже снисходительно.
async def test_grade_input_matches_any_no_variant(fresh_db):
    uid, _ = await seed_user()
    no = "hovedlemma"
    db = await _conn()
    try:
        # у слова несколько норвежских вариантов в translate.no
        data = json.dumps({"translate": {"no": [no, "bivariant"], "ru": ["перевод"]},
                           "part_of_speech": "noun"})
        await db.execute("INSERT INTO word_pool (norwegian,data,level,created_at) VALUES (?,?,?,?)",
                         (no, data, "C1", _now()))
        await db.commit()
    finally:
        await _release(db)
    # ответ — второй вариант, снисходительно (регистр)
    res = await grade_placement(
        uid, "ru", [{"no": no, "level": "C1", "answer": "BIVARIANT", "type": "input"}])
    assert res["perLevel"]["C1"]["ok"] == 1, "должно засчитывать любой вариант translate.no"


# (3) grade choice по-прежнему работает: верно если answer ∈ переводам слова на lang.
async def test_grade_choice_still_works(fresh_db):
    uid, _ = await seed_user()
    no = "hus"
    ru = "дом"
    await _seed_pool_word(no, ru, level="A1", pos="noun")

    # верный перевод (регистр/пробелы не важны для choice — strip/lower)
    ok = await grade_placement(
        uid, "ru", [{"no": no, "level": "A1", "answer": "  Дом  ", "type": "choice"}])
    assert ok["perLevel"]["A1"]["ok"] == 1, "верный перевод должен засчитываться (choice)"

    # неверный перевод — не засчитывается
    bad = await grade_placement(
        uid, "ru", [{"no": no, "level": "A1", "answer": "кошка", "type": "choice"}])
    assert bad["perLevel"]["A1"]["ok"] == 0

    # ввод норвежской леммы на choice-уровне НЕ засчитывается (choice сверяется с переводом)
    norsk = await grade_placement(
        uid, "ru", [{"no": no, "level": "A1", "answer": no, "type": "choice"}])
    assert norsk["perLevel"]["A1"]["ok"] == 0, "на choice сверяем перевод, не норв. лемму"


# (4) профиль B1-B2: силён до B2, слаб на C1/C2 (на input отвечает НЕВЕРНО) → старт ~B2, не C2.
#     На верхних уровнях ученик «узнаёт» (если бы был choice — угадал бы), но ввести не может,
#     поэтому продукция честно режет уровень до B2.
async def test_profile_strong_to_b2_weak_input_starts_b2(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos=("noun",))
    uid, _ = await seed_user()

    answers = []
    full = {"A1", "A2", "B1", "B2"}
    for lv in LEVELS:
        items = list(by_level[lv].items())[:8]   # (ru, no)
        is_input = lv in PLACEMENT_INPUT_LEVELS
        qtype = "input" if is_input else "choice"
        for ru, no in items:
            if lv in full:
                # верный ответ: choice → перевод, input(B2) → норв. лемма
                ans = no if is_input else ru
            else:
                # C1/C2 — на ввод ответить не может (вводит мусор/перевод — всё неверно)
                ans = "__не_могу_ввести__"
            answers.append({"no": no, "level": lv, "answer": ans, "type": qtype})

    res = await grade_placement(uid, "ru", answers)
    assert res["level"] == "B2", f"ожидали старт B2 (не C2), получили {res['level']}"
    assert await get_start_level(uid) == "B2"
    # перепроверим логику профиля: B2 сдан (input верный), C1/C2 — провалены
    assert res["perLevel"]["B2"]["ok"] == 8
    assert res["perLevel"]["C1"]["ok"] == 0 and res["perLevel"]["C2"]["ok"] == 0
