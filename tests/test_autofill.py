"""Покрытие autofill.py с ЗАМОКАННЫМ LLM (ask_json): генерация/обогащение слов и восстановление «ё»
работают без реального Gemini. Это сеть под дробление autofill.py (бизнес-логика воркеров —
персист, маппинг — проверяется на канонических ответах модели).
"""
import pytest
import autofill
from db.pool import get_pool_id, get_pool_by_id


@pytest.fixture
def mock_ask(monkeypatch):
    """Подменить autofill.ask_json фиксированным ответом модели."""
    def _set(resp):
        async def fake(system, user, schema, **kw):
            return resp
        monkeypatch.setattr(autofill, "ask_json", fake)
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
