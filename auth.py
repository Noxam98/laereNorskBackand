import os
import re
import json
import asyncio
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import bcrypt
import jwt
from db import (
    get_user, create_user, set_user_theme, set_user_game_prefs, set_user_current_dict,
    get_user_by_google_sub, create_google_user, set_user_google, clear_user_google,
    set_user_password,
)
from models import (
    UserAuth, Token, RefreshRequest, ThemeBody, GamePrefsBody, CurrentDictBody, GoogleAuth, PasswordBody,
)

SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))
# Google OAuth: Client ID веб-приложения (из Google Cloud Console). Пусто = вход через Google выключен.
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

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


async def _verify_google(credential: str) -> dict:
    """Проверить ID-token Google (подпись по ключам Google, aud = наш Client ID, issuer).
    Возвращает payload {sub, email, email_verified, name, ...}. Бросает 401/503 при проблеме."""
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=503, detail="Google auth is not configured")
    # google-auth и его requests-транспорт тащим лениво — не нужны при парольном входе.
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    try:
        # verify_oauth2_token делает синхронный HTTP к Google за сертификатами (кешируются) — в тред
        info = await asyncio.to_thread(
            google_id_token.verify_oauth2_token, credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    if not info.get("sub") or not info.get("email_verified"):
        raise HTTPException(status_code=401, detail="Google email not verified")
    return info


async def _unique_username(email: str) -> str:
    """Сгенерировать свободный username из локальной части email (с дедупом суффиксом)."""
    base = re.sub(r"[^a-z0-9_.-]", "", (email.split("@")[0] or "").lower()) or "user"
    username, n = base, 1
    while await get_user(username):
        n += 1
        username = f"{base}{n}"
    return username


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


@router.post("/auth/google", response_model=Token)
async def google_login(body: GoogleAuth):
    """Вход/регистрация через Google. Если sub уже привязан — входим в тот аккаунт,
    иначе создаём новый (username из email). Привязка к существующему паролю — отдельно,
    через /me/link_google в настройках."""
    info = await _verify_google(body.credential)
    sub, email = info["sub"], info.get("email", "")
    user = await get_user_by_google_sub(sub)
    if not user:
        username = await _unique_username(email)
        res = await create_google_user(username, email, sub)
        if res.get("error"):
            raise HTTPException(status_code=409, detail="Could not create account")
        user = await get_user(username)
    return _token_pair(user["username"])


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
        "email": user.get("email"),
        "googleLinked": bool(user.get("google_sub")),
        "hasPassword": bool((user.get("password") or "").strip()),
    }


@router.post("/me/link_google")
async def link_google(body: GoogleAuth, user=Depends(get_current_user)):
    """Привязать Google к текущему (парольному) аккаунту из настроек."""
    info = await _verify_google(body.credential)
    sub, email = info["sub"], info.get("email", "")
    other = await get_user_by_google_sub(sub)
    if other and other["id"] != user["id"]:
        raise HTTPException(status_code=409, detail="This Google account is already linked to another user")
    if user.get("google_sub") and user["google_sub"] != sub:
        raise HTTPException(status_code=409, detail="Account already linked to a different Google account")
    await set_user_google(user["id"], sub, email)
    return {"googleLinked": True, "email": email}


@router.post("/me/set_password")
async def set_password(body: PasswordBody, user=Depends(get_current_user)):
    """Задать/сменить пароль (в т.ч. первый пароль для Google-аккаунта)."""
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    hashed = await asyncio.to_thread(hash_password, body.password)
    await set_user_password(user["id"], hashed)
    return {"hasPassword": True}


@router.post("/me/unlink_google")
async def unlink_google(user=Depends(get_current_user)):
    """Отвязать Google. Нельзя, если это единственный способ входа (нет пароля)."""
    if not (user.get("password") or "").strip():
        raise HTTPException(status_code=400, detail="Set a password before unlinking Google")
    await clear_user_google(user["id"])
    return {"googleLinked": False}


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
