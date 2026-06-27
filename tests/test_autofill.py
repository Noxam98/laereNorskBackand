"""Покрытие autofill.py с ЗАМОКАННЫМ LLM (ask_json): генерация/обогащение слов и восстановление «ё»
работают без реального Gemini. Это сеть под дробление autofill.py (бизнес-логика воркеров —
персист, маппинг — проверяется на канонических ответах модели).
"""
import pytest
import autofill
from db.pool import get_pool_id, get_pool_by_id, get_pool_meta
from db.core import normalize_word


async def _seed(no, ru, pos="noun"):
    return await autofill.get_or_create_pool(no, {
        "word": no, "translate": {"ru": [ru]}, "part_of_speech": pos})


@pytest.fixture
def mock_ask(monkeypatch):
    """Подменить ask_json фиксированным ответом модели — в обоих модулях
    (batch-процессоры живут в autofill, генерация слов — в autofill_wordgen)."""
    import autofill_wordgen
    import autofill_enrich
    def _set(resp):
        async def fake(system, user, schema, **kw):
            return resp
        for m in (autofill, autofill_wordgen, autofill_enrich):
            monkeypatch.setattr(m, "ask_json", fake)
    return _set


async def test_generate_set_words_persists(fresh_db, mock_ask):
    mock_ask({"words": [
        {"word": "hund", "translate": {"ru": ["собака"]}, "part_of_speech": "noun", "level": "A1"},
        {"word": "katt", "translate": {"ru": ["кошка"]}, "part_of_speech": "noun", "level": "A1"},
    ]})
    pids = await autofill.generate_set_words("животные", "A1", 5, "ru")
    assert len(pids) == 2
    assert await get_pool_id("hund") in pids and await get_pool_id("katt") in pids


async def test_words_from_list_persists(fresh_db, mock_ask):
    mock_ask({"words": [{"word": "bil", "translate": {"ru": ["машина"]}, "part_of_speech": "noun", "level": "A1"}]})
    pids = await autofill.words_from_list(["bil"], "ru")
    assert len(pids) == 1
    by = await get_pool_by_id(pids[0])
    assert by and by["data"]["translate"]["ru"] == ["машина"]


async def test_words_from_image_ocr(fresh_db, mock_ask):
    # words_from_image возвращает ТОЛЬКО список слов из ответа vision-модели
    mock_ask({"words": ["eple", "eple", "  ", "melk"]})   # с дублём и пустым — должны очиститься
    words = await autofill.words_from_image("ZmFrZQ==", "image/jpeg")
    assert words == ["eple", "melk"]


async def test_restore_yo_strict(mock_ask):
    # принимаем правку только если отличие РОВНО е→ё (защита от посторонних изменений модели)
    mock_ask({"items": [
        {"src": "елка", "fixed": "ёлка"},      # валидно
        {"src": "дом", "fixed": "дом"},        # без изменений → не в карте
        {"src": "кот", "fixed": "кошка"},      # НЕ е→ё → отвергаем
    ]})
    out = await autofill.restore_yo(["елка", "дом", "кот"])
    assert out == {"елка": "ёлка"}


async def test_translate_batch(mock_ask):
    mock_ask({"results": [{"word": "hund", "ru": ["собака"], "en": ["dog"]}]})
    out = await autofill.translate_batch(["hund"])
    assert normalize_word("hund") in out and out[normalize_word("hund")]["ru"] == ["собака"]


async def test_classify_batch(fresh_db, mock_ask):
    pid = await _seed("ku", "корова")
    mock_ask({"results": [{"word": "ku", "level": "A1", "topics": ["animals"]}]})
    assert await autofill.classify_batch([{"word": "ku", "translate": {"ru": ["корова"]}, "id": pid}]) == 1
    meta = await get_pool_meta("ku")
    assert meta and meta.get("level") == "A1"


async def test_describe_batch(fresh_db, mock_ask):
    pid = await _seed("sau", "овца")
    mock_ask({"results": [{"word": "sau", "ru": "овца — животное", "en": "a sheep"}]})
    assert await autofill.describe_batch([{"word": "sau", "translate": {"ru": ["овца"]}, "id": pid}]) == 1
    by = await get_pool_by_id(pid)
    assert by["description"] and by["description"].get("ru")


async def test_forms_batch(fresh_db, mock_ask):
    pid = await _seed("bok", "книга")
    mock_ask({"results": [{"word": "bok", "gender": "ei", "def_sg": "boka", "indef_pl": "bøker", "def_pl": "bøkene"}]})
    assert await autofill.forms_batch("noun", [(pid, "bok", {"translate": {"ru": ["книга"]}})]) == 1
    by = await get_pool_by_id(pid)
    assert by["forms"] and by["forms"].get("pos") == "noun" and by["forms"].get("def_sg") == "boka"


async def test_ai_game_words(fresh_db, mock_ask):
    mock_ask({"words": [{"word": "fisk", "translate": {"ru": ["рыба"]}, "part_of_speech": "noun", "level": "A1"}]})
    words = await autofill.ai_game_words("ru", "A1", "еда", 5)
    assert isinstance(words, list)
    assert await get_pool_id("fisk") is not None
