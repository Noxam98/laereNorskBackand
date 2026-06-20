"""Раздел «Учёба»: интервальные повторения над всеми словами пользователя,
статусы/фильтры, Smart Review, статистика прогресса.
Новые слова система ДОСЫПАЕТ сама (авто-добор в build_session при пустом пуле новых) —
ручного эндпоинта «докинуть» больше нет."""
from fastapi import APIRouter, Depends
from auth import get_current_user
from activity import mark_activity
from db import (
    learning_get, learning_stats, learning_due, learning_answer, learning_set_status,
    learning_placement, learning_grade, learning_activity, learning_set_level, learning_seed_starter,
    learning_session,
    learning_gate_status, learning_gate_exam, learning_gate_grade,
    learning_audit, learning_audit_grade,
)
from models import LearningAnswer, LearningStatusBody, PlacementBody, LevelBody, GateExamBody, AuditBody

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


@router.get("/learning/session")
async def learning_session_route(size: int = 20, user=Depends(get_current_user)):
    """Программа занятия от системы: каждое слово со СЛЕДУЮЩЕЙ ступенью рампы
    (mode/direction/step). Режим не выбирает игрок — приоритеты ведёт сервер."""
    return await learning_session(user["id"], size=max(1, min(50, size)))


@router.get("/learning/activity")
async def learning_activity_route(days: int = 119, user=Depends(get_current_user)):
    return await learning_activity(user["id"], days=max(7, min(370, days)))


@router.post("/learning/answer")
async def learning_answer_route(body: LearningAnswer, user=Depends(get_current_user)):
    mark_activity()
    return await learning_answer(user["id"], body.pool_id, body.correct, body.elapsed, body.mode, body.direction)


@router.get("/learning/placement")
async def learning_placement_route(lang: str = "ru", per: int = 8, user=Depends(get_current_user)):
    return await learning_placement(lang=lang, per=max(2, min(10, per)))


@router.post("/learning/placement")
async def learning_grade_route(body: PlacementBody, user=Depends(get_current_user)):
    res = await learning_grade(user["id"], body.lang, body.answers)
    # стартовый набор под определённый уровень — чтобы было что учить сразу
    seed = await learning_seed_starter(user["id"], res.get("level"))
    return {**res, "seeded": seed.get("seeded", 0)}


@router.post("/learning/level")
async def learning_level_route(body: LevelBody, user=Depends(get_current_user)):
    """Самооценка уровня (self-report): сохранить как стартовый, уточним по сессиям."""
    await learning_set_level(user["id"], body.level)
    seed = await learning_seed_starter(user["id"], body.level)
    return {"ok": True, "level": body.level, "seeded": seed.get("seeded", 0)}


@router.get("/learning/gate")
async def learning_gate_route(user=Depends(get_current_user)):
    """Состояние ворот зачётного экзамена: {pack, threshold, open}."""
    return await learning_gate_status(user["id"])


@router.get("/learning/gate/exam")
async def learning_gate_exam_route(lang: str = "ru", user=Depends(get_current_user)):
    """Собрать выборку вопросов зачётного экзамена (до SAMPLE из несданной пачки)."""
    return await learning_gate_exam(user["id"], lang=lang)


@router.post("/learning/gate/exam")
async def learning_gate_grade_route(body: GateExamBody, user=Depends(get_current_user)):
    """Оценить зачётный экзамен: сдал → сертификация пачки; провал → демоут промахов ×2."""
    mark_activity()
    return await learning_gate_grade(user["id"], body.answers, lang=body.lang)


@router.get("/learning/audit")
async def learning_audit_route(lang: str = "ru", user=Depends(get_current_user)):
    """Собрать аудит-выборку забывания (до AUDIT_CAP самых просроченных сертифицированных)."""
    return await learning_audit(user["id"], lang=lang)


@router.post("/learning/audit")
async def learning_audit_grade_route(body: AuditBody, user=Depends(get_current_user)):
    """Оценить аудит: верно → срок дальше; забыл → де-сертификация и слово назад в учёбу."""
    mark_activity()
    return await learning_audit_grade(user["id"], body.answers, lang=body.lang)


@router.post("/learning/{pool_id}/status")
async def learning_status_route(pool_id: int, body: LearningStatusBody, user=Depends(get_current_user)):
    return await learning_set_status(user["id"], pool_id, body.action)
