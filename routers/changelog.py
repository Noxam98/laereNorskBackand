"""Ченжлог: приём записей от стража пуша (админ) + выдача «что нового» юзерам."""
from fastapi import APIRouter, Depends

from auth import get_current_user, get_admin_user
from db.changelog import add_changelog, get_changelog
from models import ChangelogIngest

router = APIRouter()


@router.post("/admin/changelog")
async def changelog_ingest_route(body: ChangelogIngest, user=Depends(get_admin_user)):
    """Пачка записей одного пуша от pre-push стража (Claude Sonnet). Идемпотентно по (repo, source)."""
    return await add_changelog(body.repo, body.source, [e.model_dump() for e in body.entries])


@router.get("/changelog")
async def changelog_route(limit: int = 30, user=Depends(get_current_user)):
    """Свежие записи «что нового» (новые первыми); i18n целиком — язык выбирает фронт."""
    return {"entries": await get_changelog(limit)}
