"""D2: auth — security-граница всего API, раньше без тестов (тесты подставляли user={id} мимо auth).
JWT roundtrip + просрочка, bcrypt (+усечение 72 байт), админ-гейт, get_current_user на живой БД."""
from datetime import timedelta

import pytest
from fastapi import HTTPException

import auth
from auth import (create_token, get_current_user, hash_password, verify_password, is_admin)
from tests.conftest import seed_user


# ---------------- пароли (bcrypt) ----------------
def test_password_roundtrip_and_wrong():
    h = hash_password("hunter2")
    assert verify_password("hunter2", h) is True
    assert verify_password("wrong", h) is False


def test_password_72_byte_truncation():
    # bcrypt режет на 72 байтах — 72 одинаковых + разный хвост считаются тем же паролем
    base = "a" * 72
    h = hash_password(base)
    assert verify_password(base + "EXTRA", h) is True     # хвост за 72 байта игнорируется
    assert verify_password("a" * 71 + "b", h) is False    # различие ДО 72 — уже другой пароль


# ---------------- админ-гейт ----------------
def test_is_admin(monkeypatch):
    monkeypatch.setattr(auth, "ADMIN_USERS", {"boss"})
    assert is_admin({"username": "Boss"}) is True          # регистронезависимо
    assert is_admin({"username": "rando"}) is False
    assert is_admin(None) is False


# ---------------- JWT через get_current_user (живая БД) ----------------
async def test_valid_token_returns_user(fresh_db):
    uid, _ = await seed_user("alice")
    tok = create_token({"sub": "alice"}, timedelta(minutes=5))
    user = await get_current_user(token=tok)
    assert user["username"] == "alice" and user["id"] == uid


async def test_expired_token_401(fresh_db):
    await seed_user("bob")
    tok = create_token({"sub": "bob"}, timedelta(minutes=-1))   # уже просрочен
    with pytest.raises(HTTPException) as e:
        await get_current_user(token=tok)
    assert e.value.status_code == 401


async def test_garbage_and_unknown_user_401(fresh_db):
    with pytest.raises(HTTPException) as e:
        await get_current_user(token="not-a-jwt")
    assert e.value.status_code == 401
    tok = create_token({"sub": "ghost"}, timedelta(minutes=5))  # подпись валидна, юзера нет
    with pytest.raises(HTTPException) as e2:
        await get_current_user(token=tok)
    assert e2.value.status_code == 401
