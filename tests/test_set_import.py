"""Импорт наборов: переиспользование пула, честные счётчики и сохранение явного перевода."""

import httpx
import pytest_asyncio

from db import get_or_create_pool
from tests.conftest import seed_user


def _mk_client(uid):
    import main
    from auth import get_current_user
    user = {"id": uid, "username": "t", "isAdmin": False, "hasPassword": True, "name": "t"}
    main.app.dependency_overrides[get_current_user] = lambda: user
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                             base_url="http://test"), main.app


@pytest_asyncio.fixture
async def client(fresh_db):
    uid, did = await seed_user("importer")
    c, app = _mk_client(uid)
    yield c, uid, did
    await c.aclose()
    app.dependency_overrides.clear()


async def test_explicit_pairs_keep_translation_without_parse_llm(client, monkeypatch):
    import autofill_wordgen

    async def must_not_call(*_args, **_kwargs):
        raise AssertionError("чистые пары должны разбираться без LLM")

    monkeypatch.setattr(autofill_wordgen, "ask_json", must_not_call)
    c, _uid, did = client
    r = await c.post(f"/sets/{did}/parse-text", json={
        "text": "hund — собака\nvike = уступать дорогу",
    })
    assert r.status_code == 200, r.text
    assert r.json()["items"] == [
        {"word": "hund", "translation": "собака"},
        {"word": "vike", "translation": "уступать дорогу"},
    ]


async def test_existing_pool_word_skips_llm_and_reports_real_add(client, monkeypatch):
    import autofill_wordgen

    async def must_not_call(*_args, **_kwargs):
        raise AssertionError("однозначное существующее слово не должно идти в LLM")

    monkeypatch.setattr(autofill_wordgen, "ask_json", must_not_call)
    await get_or_create_pool("hund", {
        "word": "hund",
        "translate": {"no": ["hund"], "ru": ["собака"]},
        "part_of_speech": "noun",
    })
    c, _uid, did = client

    first = await c.post(f"/sets/{did}/import-words", json={
        "items": [{"word": "hund", "translation": "пёс"}],
        "lang": "ru",
    })
    assert first.status_code == 200, first.text
    assert first.json()["summary"] == {
        "requested": 1, "added": 1, "already_in_set": 0,
        "reused_from_pool": 1, "created": 0, "skipped": 0, "failed": 0,
    }
    assert first.json()["words"][0]["translate"]["ru"] == ["пёс"]

    second = await c.post(f"/sets/{did}/import-words", json={"words": ["hund"], "lang": "ru"})
    assert second.status_code == 200, second.text
    assert second.json()["summary"]["added"] == 0
    assert second.json()["summary"]["already_in_set"] == 1


async def test_provider_failure_is_partial_result_not_false_success(client, monkeypatch):
    import autofill_wordgen

    async def fail(*_args, **_kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(autofill_wordgen, "ask_json", fail)
    c, _uid, did = client
    r = await c.post(f"/sets/{did}/import-words", json={"words": ["nyord"], "lang": "ru"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["added"] == 0
    assert body["summary"]["failed"] == 1
    assert body["failed"] == [{"word": "nyord", "translation": "", "reason": "provider_failed"}]


async def test_import_deduplicates_before_processing(client, monkeypatch):
    import autofill_wordgen

    calls = 0

    async def fake(_system, _user, _schema, **_kwargs):
        nonlocal calls
        calls += 1
        return {"words": [{
            "word": "bil", "translate": {"ru": ["машина"]},
            "part_of_speech": "noun", "level": "A1",
        }]}

    monkeypatch.setattr(autofill_wordgen, "ask_json", fake)
    monkeypatch.setattr(autofill_wordgen.llm, "embed_enabled", lambda: False)
    c, _uid, did = client
    r = await c.post(f"/sets/{did}/import-words", json={
        "items": [
            {"word": "bil", "translation": ""},
            {"word": "BIL", "translation": "машина"},
        ],
        "lang": "ru",
    })
    assert r.status_code == 200, r.text
    assert calls == 1
    assert r.json()["summary"]["requested"] == 1
    assert r.json()["summary"]["added"] == 1
    word = r.json()["words"][0]
    assert word["translate"]["ru"] == ["машина"]
