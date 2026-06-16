import os
import json
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import bcrypt
import jwt
from db import get_user, create_user, set_user_theme, set_user_game_prefs, set_user_current_dict
from models import UserAuth, Token, RefreshRequest, ThemeBody, GamePrefsBody, CurrentDictBody

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Админы — список имён через env ADMIN_USERS (через запятую), регистронезависимо.
ADMIN_USERS = {u.strip().lower() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()}


def is_admin(user) -> bool:
    return bool(user) and (user.get("username", "").lower() in ADMIN_USERS)


def hash_password(p):
    return bcrypt.hashpw(p.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(p, h):
    try:
        return bcrypt.checkpw(p.encode("utf-8")[:72], h.encode("utf-8"))
    except Exception:
        return False


def create_token(data: dict, expires: timedelta):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + expires})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await get_user(username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


async def get_admin_user(user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(status_code=403, detail="Forbidden")
    return user


def _token_pair(username: str):
    sub = {"sub": username}
    return {
        "access_token": create_token(sub, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)),
        "refresh_token": create_token(sub, timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)),
        "token_type": "bearer",
    }


router = APIRouter()


@router.post("/register")
async def register(user: UserAuth):
    if not user.username.strip():
        raise HTTPException(status_code=400, detail="Username cannot be empty")
    if len(user.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    if await get_user(user.username):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed = await asyncio.to_thread(hash_password, user.password)  # bcrypt CPU — в треде
    return await create_user(user.username, hashed)


@router.post("/login", response_model=Token)
async def login(creds: UserAuth):
    user = await get_user(creds.username)
    if not user or not await asyncio.to_thread(verify_password, creds.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return _token_pair(creds.username)


@router.post("/refresh", response_model=Token)
async def refresh_token(payload: RefreshRequest = None, refresh_token: str = None):
    token = (payload.refresh_token if payload else None) or refresh_token
    if not token:
        raise HTTPException(status_code=400, detail="refresh_token is required")
    try:
        decoded = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = decoded.get("sub")
        if not await get_user(username):
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        return _token_pair(username)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Refresh token expired")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")


def _parse_game_prefs(raw):
    try:
        return json.loads(raw) if raw else None
    except Exception:
        return None


@router.get("/me")
async def me(user=Depends(get_current_user)):
    return {
        "username": user["username"], "theme": user.get("theme"), "is_admin": is_admin(user),
        "gamePrefs": _parse_game_prefs(user.get("game_prefs")),
    }


@router.post("/me/theme")
async def set_theme(body: ThemeBody, user=Depends(get_current_user)):
    theme = body.theme if body.theme in ("light", "dark") else "light"
    await set_user_theme(user["id"], theme)
    return {"theme": theme}


@router.post("/me/game_prefs")
async def set_game_prefs(body: GamePrefsBody, user=Depends(get_current_user)):
    """Запоминаем последние настройки игры (режим/направление/звук)."""
    prefs = {}
    if body.type in ("study", "input", "choice"):
        prefs["type"] = body.type
    if body.dir in ("no2int", "int2no"):
        prefs["dir"] = body.dir
    if body.sound is not None:
        prefs["sound"] = bool(body.sound)
    await set_user_game_prefs(user["id"], json.dumps(prefs, ensure_ascii=False))
    return {"gamePrefs": prefs}


@router.post("/me/current_dict")
async def set_current_dict(body: CurrentDictBody, user=Depends(get_current_user)):
    """Запоминаем последний выбранный словарь, чтобы восстановить при следующем входе."""
    await set_user_current_dict(user["id"], body.name)
    return {"currentDict": body.name}
