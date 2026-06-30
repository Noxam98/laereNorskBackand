"""homograph_split: LLM раскладывает русские переводы по части речи, запись делится на per-pos омонимы.
LLM замокан. Проверяем dry-run (план без изменений) и apply (оригинал оставлен по своей части речи,
другие части речи → отдельные записи; прочие языки целы). Одна часть речи — НЕ делим."""
import pytest
import homograph_split
from db.pool import get_or_create_pool, get_pool_by_id, get_pool_id


@pytest.fixture
def mock_llm(monkeypatch):
    def _set(resp):
        async def fake(system, user, schema, **kw):
            return resp
        monkeypatch.setattr(homograph_split, "ask_json", fake)
    return _set


async def test_split_dry_then_apply(fresh_db, mock_llm):
    pid = await get_or_create_pool("tale", {"word": "tale", "part_of_speech": "noun",
                                            "translate": {"ru": ["речь", "говорить"], "en": ["speech"]}})
    mock_llm({"results": [{"word": "tale", "noun": ["речь"], "verb": ["говорить"]}]})
    # dry-run — план есть, БД не тронута
    plan = await homograph_split.split_homographs(dry_run=True)
    assert len(plan) == 1
    assert plan[0]["keep_ru"] == ["речь"] and plan[0]["others"] == {"verb": ["говорить"]}
    by = await get_pool_by_id(pid)
    assert set(by["data"]["translate"]["ru"]) == {"речь", "говорить"}
    # apply
    await homograph_split.split_homographs(dry_run=False)
    by = await get_pool_by_id(pid)
    assert by["data"]["translate"]["ru"] == ["речь"]            # сущ. оставлено
    assert by["data"]["translate"]["en"] == ["speech"]         # другие языки целы
    vid = await get_pool_id("tale", "verb")
    assert vid and vid != pid
    assert (await get_pool_by_id(vid))["data"]["translate"]["ru"] == ["говорить"]


async def test_single_pos_not_split(fresh_db, mock_llm):
    pid = await get_or_create_pool("bil", {"word": "bil", "part_of_speech": "noun",
                                           "translate": {"ru": ["машина", "автомобиль"]}})
    mock_llm({"results": [{"word": "bil", "noun": ["машина", "автомобиль"]}]})   # обе — сущ.
    plan = await homograph_split.split_homographs(dry_run=False)
    assert plan == []                                          # одна часть речи — не делим
    assert (await get_pool_by_id(pid))["data"]["translate"]["ru"] == ["машина", "автомобиль"]
