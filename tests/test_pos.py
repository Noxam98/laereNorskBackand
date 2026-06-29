"""Нормализация частей речи (норв.↔англ.) + классификация служебных по канону."""
import json
import pytest
from pos import normalize_pos
from db.learning import is_function_word
from db.core import _conn, _release


def test_normalize_pos_norwegian_to_english():
    assert normalize_pos("substantiv") == "noun"
    assert normalize_pos("adjektiv") == "adjective"
    assert normalize_pos("preposisjon") == "preposition"
    assert normalize_pos("pronomen") == "pronoun"
    assert normalize_pos("konjunksjon") == "conjunction"
    assert normalize_pos("subjunksjon") == "conjunction"
    assert normalize_pos("tallord") == "numeral"
    assert normalize_pos("interjeksjon") == "interjection"
    assert normalize_pos("VERB") == "verb"          # регистр не важен
    assert normalize_pos("noun") == "noun"          # уже англ. — без изменений
    assert normalize_pos("") == "" and normalize_pos(None) == ""


def test_is_function_word_both_spellings():
    # служебное распознаётся И по норв., И по англ. написанию
    assert is_function_word("på", {"part_of_speech": "preposisjon"})
    assert is_function_word("på", {"part_of_speech": "preposition"})   # раньше НЕ распознавалось
    assert is_function_word("og", {"part_of_speech": "konjunksjon"})
    assert is_function_word("og", {"part_of_speech": "conjunction"})
    assert is_function_word("å", {})                                   # частица
    assert is_function_word("ikke", {"part_of_speech": "adverb"})      # функц. наречие (белый список)
    # контентное — не служебное (оба написания)
    assert not is_function_word("bil", {"part_of_speech": "substantiv"})
    assert not is_function_word("bil", {"part_of_speech": "noun"})


@pytest.mark.asyncio
async def test_set_pool_pos_updates_column_and_data(fresh_db):
    """set_pool_pos обновляет И колонку pos, И data.part_of_speech, нормализуя написание."""
    from db import set_pool_pos, get_or_create_pool
    pid = await get_or_create_pool("gå", {"part_of_speech": "", "translate": {"no": ["gå"]}})
    await set_pool_pos(pid, "verb")
    db = await _conn()
    try:
        async with db.execute("SELECT COALESCE(pos,'') p, data FROM word_pool WHERE id=?", (pid,)) as c:
            r = await c.fetchone()
    finally:
        await _release(db)
    assert r["p"] == "verb"                                        # колонка обновлена (не осталась пустой)
    assert json.loads(r["data"])["part_of_speech"] == "verb"       # и data синхронна
    # норвежское написание сводится к канону
    pid2 = await get_or_create_pool("bil", {"part_of_speech": "", "translate": {"no": ["bil"]}})
    await set_pool_pos(pid2, "substantiv")
    db = await _conn()
    try:
        async with db.execute("SELECT COALESCE(pos,'') p FROM word_pool WHERE id=?", (pid2,)) as c:
            assert (await c.fetchone())["p"] == "noun"
    finally:
        await _release(db)


@pytest.mark.asyncio
async def test_set_pool_pos_merges_on_conflict(fresh_db):
    """Если каноничная (norwegian,pos) уже есть — классификация дубля (пустой pos) сливает его в неё."""
    from db import set_pool_pos, get_or_create_pool
    canon = await get_or_create_pool("gå", {"part_of_speech": "verb", "translate": {"no": ["gå"]}})
    dup = await get_or_create_pool("gå", {"part_of_speech": "", "translate": {"no": ["gå"]}})
    assert canon != dup                                            # разные записи: (gå,verb) и (gå,'')
    await set_pool_pos(dup, "verb")                                # конфликт → слить dup в canon
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM word_pool WHERE norwegian='gå'") as c:
            assert (await c.fetchone())["c"] == 1                  # дубль слит, осталась одна запись
    finally:
        await _release(db)
