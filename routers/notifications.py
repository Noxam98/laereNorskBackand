"""Центр уведомлений + поиск людей (для передачи набора).

Уведомления общие на вырост (см. db/notifications.py): сейчас единственный тип — предложение
забрать набор. Поиск людей отдаёт МИНИМУМ (id + видимое имя) и прикрыт rate-limit'ом: это
поиск знакомого по имени/логину, а не выгрузка списка аккаунтов.
"""
from fastapi import APIRouter, Depends, HTTPException, Request
from auth import get_current_user
from activity import mark_activity
from ratelimit import _hit
from db.users import search_users
from db.notifications import list_notifications, mark_read, decide_share

router = APIRouter()

SEARCH_MAX = 60        # запросов поиска людей
SEARCH_WINDOW = 60     # …за столько секунд на пользователя


@router.get("/users/search")
async def users_search(q: str = "", user=Depends(get_current_user)):
    """Поиск людей по имени/логину (подстрока, минимум 2 символа, максимум 10 результатов).
    В ответе только id и видимое имя — ни почты, ни прогресса, ни признака онлайн."""
    _hit(f"usearch:{user['id']}", SEARCH_MAX, SEARCH_WINDOW)   # сам бросит 429 при переборе
    return {"users": await search_users(q, exclude_id=user["id"])}


@router.get("/notifications")
async def notifications_list(limit: int = 50, user=Depends(get_current_user)):
    return await list_notifications(user["id"], limit=limit)


@router.post("/notifications/read")
async def notifications_read(user=Depends(get_current_user)):
    """Панель открыли — гасим бейдж непрочитанных (решения по предложениям это не трогает)."""
    return await mark_read(user["id"])


@router.post("/notifications/{nid}/accept")
async def notifications_accept(nid: int, user=Depends(get_current_user)):
    """Принять предложенный набор — создаётся СВОЯ копия (редактируй как хочешь)."""
    mark_activity()
    res = await decide_share(user["id"], nid, True)
    if res.get("error"):
        raise HTTPException(status_code=404 if res["error"] == "Not found" else 409, detail=res["error"])
    return res


@router.post("/notifications/{nid}/decline")
async def notifications_decline(nid: int, user=Depends(get_current_user)):
    res = await decide_share(user["id"], nid, False)
    if res.get("error"):
        raise HTTPException(status_code=404 if res["error"] == "Not found" else 409, detail=res["error"])
    return res
