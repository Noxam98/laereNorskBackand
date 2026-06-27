"""Обновление переводов в пуле сбрасывает озвучку перевода (tts_tr_done=0).
Это основа фонового бэкилла буквы «ё» (yo_fix_loop) и переозвучки: поправили текст —
аудио перегенерится воркером tts_translation_loop.
"""
from db.pool import (
    get_or_create_pool, get_pool_id, update_pool_translate,
    mark_tr_tts_done, tr_tts_pending,
)


async def _pending_ids():
    return [pid for pid, _ in await tr_tts_pending(10000)]


async def test_pool_round_trip(fresh_db):
    """get_or_create_pool кладёт слово, get_pool_id находит его по норвежскому написанию."""
    pid = await get_or_create_pool("sykkel", {
        "word": "sykkel", "translate": {"ru": ["велосипед"]},
        "part_of_speech": "noun", "level": "A1"})
    assert pid
    assert await get_pool_id("sykkel") == pid


async def test_update_translate_resets_tts(fresh_db):
    """После update_pool_translate слово снова попадает в очередь озвучки переводов."""
    pid = await get_or_create_pool("bok", {
        "word": "bok", "translate": {"ru": ["книга"]},
        "part_of_speech": "noun", "level": "A1"})
    await mark_tr_tts_done(pid)                       # «озвучено»
    assert pid not in await _pending_ids()
    await update_pool_translate(pid, {"ru": ["книга", "том"]})   # поправили перевод
    assert pid in await _pending_ids()               # → снова в очереди озвучки
