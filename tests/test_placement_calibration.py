"""Калибровка входного теста (placement): тест НЕ должен завышать уровень.

Единый контракт вопроса:
- choice (A1,A2,B1): {no, level, type:"choice", options:[4 перевода на lang]} — узнавание;
- input  (B2,C1,C2): {no:<норв. лемма — ключ грейда>, prompt:<перевод>, level, type:"input"} —
  продукция (ввод нельзя угадать → честнее на верхних уровнях).

Проверяем строение вопросов и КОНСЕРВАТИВНУЮ оценку старта:
- по `per` слов на уровень (по умолчанию 8);
- порог сдачи уровня 0.8 (75% не засчитывается);
- уровень с <PLACEMENT_MIN ответов не засчитывается;
- профиль 100% A1..B2 + 30% C1/C2 → старт B2 (не C1/C2);
- input грейдится по норвежской лемме СНИСХОДИТЕЛЬНО (å→a ø→o æ→ae, срез диакритик).

Тесты осмысленны: проверяют поведение, а не подгоняются под реализацию.
"""
import json
import pytest

from db.learning import (
    build_placement, grade_placement, get_start_level,
    LEVELS, PLACEMENT_PASS, PLACEMENT_MIN, PLACEMENT_INPUT_LEVELS,
)
from db.core import _conn, _release, _now
from tests.conftest import seed_user, seed_word


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


async def _seed_full_pool(per_level=12, pos_per_level=("noun", "verb")):
    """Засеять пул так, чтобы на каждом уровне CEFR хватало слов под `per` вопросов
    и под правдоподобные дистракторы той же части речи.
    Возвращает {level: {ru_translation: norwegian}} для разбора ответов по уровням."""
    by_level = {}
    for lv in LEVELS:
        by_level[lv] = {}
        for pos in pos_per_level:
            for i in range(per_level):
                # норвежское слово — нижний регистр: get_pool_id ищет по normalize_word
                # (strip+lower) с точным сравнением, иначе разбор ответов не найдёт слово.
                no = f"{lv}_{pos}_{i}_no".lower()
                ru = f"{lv}_{pos}_{i}_ru"
                await _seed_pool_word(no, ru, level=lv, pos=pos)
                by_level[lv][ru] = no
    return by_level


# (1a) строение вопроса: 4 варианта, верный среди них, дистракторы — переводы ТОГО ЖЕ уровня
#      (а не случайные из других уровней — иначе исключение «по очевидности» завышает балл).
async def test_question_shape_and_same_level_distractors(fresh_db):
    # На каждом уровне — слова двух частей речи; переводы узнаваемы по префиксу уровня.
    by_level = await _seed_full_pool(per_level=12, pos_per_level=("noun", "verb"))
    translations_by_level = {lv: set(by_level[lv].keys()) for lv in LEVELS}
    # перевод → его уровень (для строгой проверки «дистрактор не из чужого уровня»)
    ru_to_level = {ru: lv for lv in LEVELS for ru in by_level[lv]}

    res = await build_placement(lang="ru", per=8)
    qs = res["questions"]
    assert qs, "входной тест должен сгенерировать вопросы"

    # карта правильного перевода по норвежскому слову
    correct_ru = {no: ru for lv in LEVELS for ru, no in by_level[lv].items()}

    # верхние уровни идут на ВВОД, нижние — на ВЫБОР
    assert any(q["type"] == "input" for q in qs), "на верхних уровнях должны быть вопросы-ввод"
    assert any(q["type"] == "choice" for q in qs), "на нижних уровнях должны быть вопросы-выбор"

    for q in qs:
        lv = q["level"]
        if lv in PLACEMENT_INPUT_LEVELS:
            # input: тип input, есть prompt(перевод) и no(лемма), без options
            assert q["type"] == "input", f"уровень {lv} должен быть input"
            assert "options" not in q
            assert q["prompt"] == correct_ru[q["no"]], "prompt — перевод леммы на lang"
            continue
        # choice: ровно 4 варианта, без дублей
        assert q["type"] == "choice", f"уровень {lv} должен быть choice"
        assert len(q["options"]) == 4
        assert len(set(q["options"])) == 4
        # верный перевод присутствует среди вариантов
        corr = correct_ru[q["no"]]
        assert corr in q["options"]
        # дистракторы (3 неверных) — переводы слов ТОГО ЖЕ уровня (не из других уровней)
        distractors = [o for o in q["options"] if o != corr]
        assert len(distractors) == 3
        for d in distractors:
            assert d in translations_by_level[lv], (
                f"дистрактор {d!r} не из уровня {lv} (вопрос {q['no']!r})")
            assert ru_to_level[d] == lv, (
                f"дистрактор {d!r} с уровня {ru_to_level[d]}, ожидался {lv}")


# (1b) когда на уровне ЕСТЬ кандидаты той же части речи, дистракторы — той же части речи
#      (не подмешиваются слова другой части речи). Берём пул, где у каждого уровня ОДНА часть
#      речи, поэтому любой дистрактор того же уровня неизбежно той же части речи — это и есть
#      осмысленная проверка свойства «дистракторы того же уровня той же части речи».
async def test_distractors_same_part_of_speech_when_candidates_exist(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos_per_level=("noun",))
    correct_ru = {no: ru for lv in LEVELS for ru, no in by_level[lv].items()}

    res = await build_placement(lang="ru", per=8)
    qs = res["questions"]
    assert qs

    for q in qs:
        if q["type"] != "choice":   # дистракторы есть только у вопросов-выбора
            continue
        corr = correct_ru[q["no"]]
        distractors = [o for o in q["options"] if o != corr]
        assert len(distractors) == 3
        # все слова уровня — noun, целевое — тоже noun → каждый дистрактор той же части речи
        for d in distractors:
            assert d.split("_")[1] == "noun", (
                f"дистрактор {d!r} другой части речи (есть кандидаты-noun того же уровня)")


# (2) per=8 по умолчанию: при достатке слов получаем ~ len(LEVELS) * 8 вопросов
async def test_default_per_is_8(fresh_db):
    await _seed_full_pool(per_level=12, pos_per_level=("noun",))
    res = await build_placement(lang="ru")   # per по умолчанию
    qs = res["questions"]
    # по 8 вопросов на каждый из 6 уровней
    per_lvl = {lv: 0 for lv in LEVELS}
    for q in qs:
        per_lvl[q["level"]] += 1
    for lv in LEVELS:
        assert per_lvl[lv] == 8, f"на уровне {lv} должно быть 8 вопросов, а не {per_lvl[lv]}"
    assert len(qs) == len(LEVELS) * 8


# (3) профиль 100% A1..B2 + 30% C1/C2 → старт B2 (НЕ C1/C2)
async def test_high_low_profile_caps_at_b2(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos_per_level=("noun",))
    uid, _ = await seed_user()

    # формируем ответы: 8 вопросов на уровень.
    # A1..B2 — все верные; C1/C2 — ~30% верных (примерно везучие угадывания).
    # Верный ответ зависит от типа вопроса: choice → перевод (ru), input → норвежская лемма (no).
    answers = []
    full = ["A1", "A2", "B1", "B2"]
    low = {"C1": 3, "C2": 2}   # из 8: 3/8≈38%, 2/8=25% — обе ниже 0.8
    for lv in LEVELS:
        items = list(by_level[lv].items())[:8]   # (ru, no)
        n_correct = 8 if lv in full else low[lv]
        is_input = lv in PLACEMENT_INPUT_LEVELS
        qtype = "input" if is_input else "choice"
        for i, (ru, no) in enumerate(items):
            correct = no if is_input else ru
            ans = correct if i < n_correct else "__неверный_ответ__"
            answers.append({"no": no, "level": lv, "answer": ans, "type": qtype})

    res = await grade_placement(uid, "ru", answers)
    assert res["level"] == "B2", f"ожидали B2, получили {res['level']}"
    assert await get_start_level(uid) == "B2"


# (4) порог 0.8: уровень с 75% верных НЕ засчитывается
async def test_threshold_75pct_not_passed(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos_per_level=("noun",))
    uid, _ = await seed_user()

    # A1 — 100%, A2 — ровно 75% (6 из 8) → A2 не должен засчитаться, старт остаётся A1.
    answers = []
    a1 = list(by_level["A1"].items())[:8]
    for ru, no in a1:
        answers.append({"no": no, "level": "A1", "answer": ru})
    a2 = list(by_level["A2"].items())[:8]
    for i, (ru, no) in enumerate(a2):
        ans = ru if i < 6 else "__нет__"   # 6/8 = 0.75 < 0.8
        answers.append({"no": no, "level": "A2", "answer": ans})

    # контроль предпосылки теста
    assert 6 / 8 < PLACEMENT_PASS <= 1.0

    res = await grade_placement(uid, "ru", answers)
    assert res["perLevel"]["A2"]["ok"] == 6 and res["perLevel"]["A2"]["total"] == 8
    assert res["level"] == "A1", f"75% не должно засчитывать уровень, получили {res['level']}"


# (5) уровень с <PLACEMENT_MIN отвеченных вопросов не засчитывается
async def test_too_few_answers_not_passed(fresh_db):
    by_level = await _seed_full_pool(per_level=12, pos_per_level=("noun",))
    uid, _ = await seed_user()

    # A1 — полноценно сдан (8/8). A2 — только 3 ответа, все верные (100%),
    # но 3 < PLACEMENT_MIN(=4) → уровень не засчитывается, старт остаётся A1.
    assert PLACEMENT_MIN >= 4
    answers = []
    for ru, no in list(by_level["A1"].items())[:8]:
        answers.append({"no": no, "level": "A1", "answer": ru})
    for ru, no in list(by_level["A2"].items())[:3]:   # всего 3 ответа на A2
        answers.append({"no": no, "level": "A2", "answer": ru})

    res = await grade_placement(uid, "ru", answers)
    assert res["perLevel"]["A2"]["total"] == 3 and res["perLevel"]["A2"]["ok"] == 3
    assert res["level"] == "A1", (
        f"<{PLACEMENT_MIN} ответов не должно засчитывать уровень, получили {res['level']}")


# (6) input грейдится по НОРВЕЖСКОЙ лемме СНИСХОДИТЕЛЬНО: lower/trim + å→a ø→o æ→ae + срез
#     диакритик. Ввод перевода (как для choice) на input-уровне НЕ засчитывается — продукция честнее.
async def test_input_graded_leniently_against_norwegian_lemma(fresh_db):
    uid, _ = await seed_user()
    # слово B2 с диакритиками в лемме; перевод на ru — для контроля «не засчитываем перевод»
    no = "spørsmål"   # норвежская лемма
    ru = "вопрос"
    await _seed_pool_word(no, ru, level="B2", pos="noun")

    # снисходительные написания леммы должны проходить, перевод (ru) — нет
    cases = [
        ("spørsmål", True),    # точное совпадение
        ("  Spørsmål ", True), # регистр/пробелы
        ("sporsmal", True),    # ø→o, å→a (foldLoose)
        ("SPORSMAL", True),    # + регистр
        (ru, False),           # перевод на input-уровне не засчитывается
        ("__мимо__", False),
    ]
    for answer, expect_ok in cases:
        res = await grade_placement(
            uid, "ru", [{"no": no, "level": "B2", "answer": answer, "type": "input"}])
        got_ok = res["perLevel"]["B2"]["ok"] == 1
        assert got_ok == expect_ok, f"ответ {answer!r}: ожидали ok={expect_ok}, получили ok={got_ok}"


# (7) тип выводится из уровня, если клиент его не прислал (обратная совместимость):
#     верхний уровень без type → грейдится как input (по лемме), а не как choice.
async def test_type_inferred_from_level_when_missing(fresh_db):
    uid, _ = await seed_user()
    no = "baerekraft"  # ascii-лемма, чтобы перевод и лемма точно различались
    ru = "устойчивость"
    await _seed_pool_word(no, ru, level="C1", pos="noun")
    # без type: C1 — input-уровень → засчитываем ввод леммы, а перевод не засчитываем
    r1 = await grade_placement(uid, "ru", [{"no": no, "level": "C1", "answer": no}])
    assert r1["perLevel"]["C1"]["ok"] == 1
    r2 = await grade_placement(uid, "ru", [{"no": no, "level": "C1", "answer": ru}])
    assert r2["perLevel"]["C1"]["ok"] == 0
