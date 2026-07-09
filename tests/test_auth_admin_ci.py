"""Регрессия на эскалацию привилегий: username CI-уникальны, поэтому регистро-вариант админского
имени нельзя зарегистрировать (иначе is_admin по .lower() отдал бы права чужому аккаунту)."""
import pytest

from db import create_user, username_taken_ci
from auth import register, is_admin
from models import UserAuth


async def test_username_taken_ci_matches_any_case(fresh_db):
    await create_user("Maksym", "hash")
    assert await username_taken_ci("maksym") is True     # тот же аккаунт по CI
    assert await username_taken_ci("MAKSYM") is True
    assert await username_taken_ci("MaKsYm") is True
    assert await username_taken_ci("maksym2") is False    # другое имя — свободно


async def test_register_rejects_case_variant(fresh_db):
    await create_user("Maksym", "hash")                   # «настоящий» аккаунт (в проде — админ)
    with pytest.raises(Exception) as ei:                  # HTTPException 400 "already exists"
        await register(UserAuth(username="maksym", password="secret6"))
    assert getattr(ei.value, "status_code", None) == 400
    # и наоборот — совершенно другое имя регистрируется штатно
    res = await register(UserAuth(username="someoneelse", password="secret6"))
    assert res.get("user_id")


async def test_is_admin_only_exact_lowercased_membership(monkeypatch):
    # при CI-уникальности имён .lower()-матч безопасен: сюда попадает только легитимный админ
    monkeypatch.setattr("auth.ADMIN_USERS", {"maksym"})
    assert is_admin({"username": "Maksym"}) is True
    assert is_admin({"username": "maksym"}) is True
    assert is_admin({"username": "someoneelse"}) is False
    assert is_admin(None) is False
