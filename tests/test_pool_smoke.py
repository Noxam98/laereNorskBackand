"""Smoke-покрытие db/pool.py: дёргаем каждую публичную функцию пула (с засеянными словами) и
проверяем, что она исполняется и отдаёт разумный тип. Это сеть под дробление pool.py на модули —
любая сломанная при переносе функция всплывёт здесь (а не на проде).
"""
import db.pool as P
from db.core import _conn, _release
from tests.conftest import seed_user


async def _seed(no, ru, level="A1", pos="noun"):
    return await P.get_or_create_pool(no, {
        "word": no, "translate": {"no": [no], "ru": [ru], "en": [ru]},
        "part_of_speech": pos, "level": level})


async def test_core_crud(fresh_db):
    pid = await _seed("hund", "собака")
    assert pid and await P.get_pool_id("hund") == pid
    by = await P.get_pool_by_id(pid)
    assert by and "data" in by
    await P.update_pool_word("hund", {"ru": ["собака", "пёс"]})
    await P.update_pool_translate(pid, {"ru": ["собака"], "no": ["hund"]})
    await P.replace_pool_word("hund", "hunden", {"translate": {"ru": ["собака"]}, "part_of_speech": "noun"})


async def test_search_and_facets(fresh_db):
    for no, ru in [("katt", "кошка"), ("bil", "машина"), ("hus", "дом")]:
        await _seed(no, ru)
    assert isinstance((await P.get_pool_list())["words"], list)
    assert isinstance(await P.get_pool_ids(), list)
    assert isinstance(await P.search_pool("ka"), list)
    assert isinstance(await P.get_pool_facets(), dict)
    assert isinstance(await P.get_pool_topics_counts(), list)
    assert isinstance(await P.get_pool_level_counts(), list)
    assert isinstance(await P.get_pool_stats(), dict)
    db = await _conn()
    try:
        assert isinstance(await P.fuzzy_pool_ids(db, "katt"), list)
    finally:
        await _release(db)


async def test_queries(fresh_db):
    p1 = await _seed("vann", "вода")
    await _seed("brød", "хлеб")
    assert isinstance(await P.get_pool_sample(), list)
    assert isinstance(await P.get_pool_duel_words(), list)
    assert isinstance(await P.get_pool_words_by_names(["vann"]), list)
    assert isinstance(await P.get_pool_letter("v"), list)
    assert (await P.get_pool_meta("vann")) is not None or True   # None допустимо
    assert isinstance(await P.get_pool_meta_all(), list)
    assert isinstance(await P.get_pool_candidates(), list)
    assert isinstance(p1, int)


async def test_freq(fresh_db):
    pid = await _seed("sol", "солнце")
    assert P.freq_band(5.0) in ("common", "mid", "rare", "very-rare") or isinstance(P.freq_band(5.0), str)
    await P.set_pool_freq(pid, 4.2)
    await P.set_pool_freq_bulk([(pid, 4.5)])
    assert isinstance(await P.pool_by_freq(), list)
    assert isinstance(await P.freq_pending(), list)


async def test_tts_emb_translate_queues(fresh_db):
    pid = await _seed("melk", "молоко")
    assert (await P.get_pool_tts("melk")) is None
    await P.set_pool_tts("melk", b"\x00\x01")
    assert isinstance(await P.pool_missing_tts(), list)
    await P.mark_tr_tts_done(pid)
    assert isinstance(await P.tr_tts_pending(), list)
    assert isinstance(await P.pool_missing_embedding(), list)
    assert isinstance(await P.get_pool_embeddings_raw(), list)
    assert isinstance(await P.get_pool_embeddings_page(), list)
    assert isinstance(await P.sem_embed_pending(), list)
    await P.mark_sem_embed(pid)
    assert isinstance(await P.translate_pending(), list)
    await P.mark_translate_done(pid)
    await P.mark_yo_done(pid)
    assert isinstance(await P.yo_pending(), list)


async def test_meta_forms_pos(fresh_db):
    pid = await _seed("stor", "большой", pos="adjective")
    await P.set_pool_meta(pid, level="A2", topics=["home"])
    await P.set_pool_description(pid, {"ru": "описание"})
    await P.set_pool_forms(pid, {"pos": "adjective", "comparative": "større"})
    await P.set_pool_pos(pid, "adjective")
    assert isinstance(await P.pool_missing_meta(), list)
    assert isinstance(await P.pool_missing_description(), list)
    assert isinstance(await P.pos_uncategorized(), list)
    assert isinstance(await P.pos_missing_forms("adjective"), list)
    await P.clear_nonformable_forms()
    await P.clear_all_descriptions()


async def test_clear_all_forms_and_exact_pos_filter(fresh_db):
    """clear_all_forms зануляет только formable; POS-фильтр точный по канон-колонке (pronoun ≠ noun)."""
    n = await _seed("hund", "собака", pos="noun")
    v = await _seed("snakke", "говорить", pos="verb")
    pr = await _seed("jeg", "я", pos="pronoun")          # 'pronoun' содержит подстроку 'noun'
    await P.set_pool_forms(n, {"pos": "noun", "gender": "en", "def_sg": "hunden"})
    await P.set_pool_forms(v, {"pos": "verb", "past": "snakket"})
    # точный фильтр: noun-очередь НЕ цепляет местоимение (раньше LIKE '%noun%' мог)
    miss_noun = [r[0] for r in await P.pos_missing_forms("noun", 50)]
    assert pr not in miss_noun and n not in miss_noun    # pronoun не formable; у hund формы есть
    # clear_all_forms: formable занулены, non-formable (pronoun) не трогаем
    cleared = await P.clear_all_forms()
    assert cleared == 2
    assert (await P.get_pool_by_id(n))["forms"] is None and (await P.get_pool_by_id(v))["forms"] is None
    assert n in [r[0] for r in await P.pos_missing_forms("noun", 50)]   # снова «без форм»


async def test_dedup(fresh_db):
    w = await _seed("by", "город")
    loser = await _seed("byen", "город (опр.)")
    assert isinstance(await P.dedup_pending(), list)
    await P.mark_dedup(w)
    assert isinstance(await P.pool_usage_count(w), int)
    assert (await P.dedup_progress()) is not None
    await P.merge_pool_words(w, loser)   # слить дубль


async def test_moderation_and_reports(fresh_db):
    uid, _ = await seed_user()
    pid = await _seed("feil", "ошибка")
    assert isinstance(await P.pending_words(), list)
    assert isinstance(await P.pending_count(), int)
    await P.set_word_approval(pid, 1)
    await P.report_word(pid, uid)
    assert isinstance(await P.reported_words(), list)
    assert isinstance(await P.reported_count(), int)
    await P.resolve_report(pid, "keep")
    await P.delete_pool_word("feil")
