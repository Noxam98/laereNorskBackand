"""Актуальные Gemini-профили и совместимость запроса с API моделей 3.6/3.5 Lite."""
from types import SimpleNamespace

import autofill_wordgen
from llm import client
from llm.settings import USER_TEXT_MODEL


def test_latest_model_profiles():
    assert USER_TEXT_MODEL == "gemini-3.5-flash-lite"
    assert autofill_wordgen.SET_GEN_MODELS == ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
    assert autofill_wordgen.VISION_MODELS == ["gemini-3.6-flash"]


async def test_latest_models_do_not_receive_sampling_parameters(monkeypatch):
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake = SimpleNamespace(
        with_options=lambda **_kwargs: fake,
        chat=SimpleNamespace(completions=Completions()),
    )

    async def no_usage(*_args):
        return None

    monkeypatch.setattr(client, "get_client", lambda: fake)
    monkeypatch.setattr(client.quota, "text_candidates",
                        lambda _purpose, _model: [("gemini-3.6-flash", "key", 0)])
    monkeypatch.setattr(client.quota, "incr_text", no_usage)

    result = await client.ask_json(
        "system", "user",
        {"name": "answer", "schema": {"type": "object"}},
        temperature=0, max_tokens=128,
    )

    assert result == {"ok": True}
    assert calls[0]["model"] == "gemini-3.6-flash"
    assert "temperature" not in calls[0]
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_tokens"] == 128


async def test_legacy_or_custom_model_keeps_temperature(monkeypatch):
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            message = SimpleNamespace(content='{"ok": true}')
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake = SimpleNamespace(
        with_options=lambda **_kwargs: fake,
        chat=SimpleNamespace(completions=Completions()),
    )

    async def no_usage(*_args):
        return None

    monkeypatch.setattr(client, "get_client", lambda: fake)
    monkeypatch.setattr(client.quota, "text_candidates",
                        lambda _purpose, _model: [("custom-model", "key", 0)])
    monkeypatch.setattr(client.quota, "incr_text", no_usage)

    await client.ask_json(
        "system", "user",
        {"name": "answer", "schema": {"type": "object"}},
        temperature=0,
    )

    assert calls[0]["temperature"] == 0
    assert "reasoning_effort" not in calls[0]
