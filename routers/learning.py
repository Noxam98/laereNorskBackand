"""Раздел «Учёба»: интервальные повторения над всеми словами пользователя,
статусы/фильтры, Smart Review, статистика прогресса и «докинуть слов» по уровню."""
from fastapi import APIRouter, Depends
from auth import get_current_user
from activity import mark_activity
from db import (
    learning_get, learning_stats, learning_due, learning_answer, learning_set_status, learning_suggest,
    learning_placement, learning_grade,
)
from models import LearningAnswer, LearningStatusBody, SuggestBody, PlacementBody

router = APIRouter()


@router.get("/learning")
async def learning_list(status: str = None, level: str = None, topic: str = None, q: str = None,
                        sort: str = "strength", limit: int = 200, offset: int = 0,
                        user=Depends(get_current_user)):
    return await learning_get(user["id"], status=status, level=level, topic=topic, q=q,
                              sort=sort, limit=limit, offset=offset)


@router.get("/learning/stats")
async def learning_stats_route(user=Depends(get_current_user)):
    return await learning_stats(user["id"])


@router.get("/learning/due")
async def learning_due_route(limit: int = 20, user=Depends(get_current_user)):
    return await learning_due(user["id"], limit=limit)


@router.post("/learning/answer")
async def learning_answer_route(body: LearningAnswer, user=Depends(get_current_user)):
    mark_activity()
    return await learning_answer(user["id"], body.pool_id, body.correct, body.elapsed, body.mode)


@router.get("/learning/placement")
async def learning_placement_route(lang: str = "ru", per: int = 4, user=Depends(get_current_user)):
    return await learning_placement(lang=lang, per=max(2, min(8, per)))


@router.post("/learning/placement")
async def learning_grade_route(body: PlacementBody, user=Depends(get_current_user)):
    return await learning_grade(user["id"], body.lang, body.answers)


@router.post("/learning/{pool_id}/status")
async def learning_status_route(pool_id: int, body: LearningStatusBody, user=Depends(get_current_user)):
    return await learning_set_status(user["id"], pool_id, body.action)


@router.post("/learning/suggest")
async def learning_suggest_route(body: SuggestBody, user=Depends(get_current_user)):
    mark_activity()
    count = max(1, min(50, body.count or 10))
    return await learning_suggest(user["id"], count=count, level=(body.level or None))
