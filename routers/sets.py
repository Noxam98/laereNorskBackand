"""Личные наборы для изучения слов («sets»).

Набор = словарь пользователя с hidden=0 (скрытый авто-словарь hidden=1 сюда не попадает).
Прогресс SRS ОБЩИЙ по слову (user_words/pool_id) — набор лишь группирует слова (dict_words)
и несёт флаг studying: вкл → слова набора питают ежедневную умную сессию «Сегодня»;
выкл → слова учатся только явно, кнопкой «Учить набор» (GET /sets/{id}/session).
CRUD — тонкие обёртки над db.dictionaries (логика жива с Фазы 1, в Фазе 2 убрали лишь роуты)."""
from fastapi import APIRouter, Depends, HTTPException
from auth import get_current_user
from activity import mark_activity
from db import (
    list_user_sets, create_dictionary, rename_dictionary, delete_dictionary,
    set_dictionary_studying, add_words_to_set, remove_word_from_set, get_set_words,
    sets_for_words, reset_set_ramp, learning_session, get_pool_id,
)

router = APIRouter()


def _bad(res):
    """db-функции возвращают {"error": ...} — превращаем в 400/404."""
    if isinstance(res, dict) and res.get("error"):
        err = res["error"]
        code = 404 if err == "Not found" else 400
        raise HTTPException(status_code=code, detail=err)
    return res


@router.get("/sets")
async def sets_list(user=Depends(get_current_user)):
    """Личные наборы пользователя: [{id, name, studying, count}]."""
    return await list_user_sets(user["id"])


@router.post("/sets")
async def sets_create(body: dict, user=Depends(get_current_user)):
    """Создать набор {name}. По умолчанию studying=0 (учить — кнопкой/тогглом)."""
    return _bad(await create_dictionary(user["id"], (body or {}).get("name", "")))


@router.post("/sets/membership")
async def sets_membership(body: dict, user=Depends(get_current_user)):
    """Карта pool_id → [set_id] для пикера «Добавить в набор»: где уже лежит каждое слово."""
    pool_ids = (body or {}).get("pool_ids") or []
    return await sets_for_words(user["id"], pool_ids)


@router.patch("/sets/{set_id}")
async def sets_rename(set_id: int, body: dict, user=Depends(get_current_user)):
    """Переименовать набор {name}."""
    return _bad(await rename_dictionary(user["id"], set_id, (body or {}).get("name", "")))


@router.delete("/sets/{set_id}")
async def sets_delete(set_id: int, user=Depends(get_current_user)):
    """Удалить набор (членство чистится; прогресс SRS слов остаётся)."""
    return _bad(await delete_dictionary(user["id"], set_id))


@router.post("/sets/{set_id}/studying")
async def sets_studying(set_id: int, body: dict, user=Depends(get_current_user)):
    """Тоггл «питать ежедневную учёбу» для набора."""
    return _bad(await set_dictionary_studying(user["id"], set_id, bool((body or {}).get("studying"))))


@router.get("/sets/{set_id}/words")
async def sets_words(set_id: int, user=Depends(get_current_user)):
    """Слова набора: [{pool_id, norwegian, translate, part_of_speech, level}]."""
    words = await get_set_words(user["id"], set_id)
    if words is None:
        raise HTTPException(status_code=404, detail="Not found")
    return {"words": words}


@router.post("/sets/{set_id}/words")
async def sets_words_add(set_id: int, body: dict, user=Depends(get_current_user)):
    """Добавить слова в набор: {pool_ids: [...]} или {pool_id} / {norwegian}."""
    mark_activity()
    body = body or {}
    pool_ids = list(body.get("pool_ids") or [])
    if not pool_ids:
        pid = body.get("pool_id")
        if not pid:
            word = (body.get("norwegian") or body.get("word") or "").strip()
            pid = await get_pool_id(word) if word else None
        if pid:
            pool_ids = [pid]
    if not pool_ids:
        raise HTTPException(status_code=400, detail="No words")
    return _bad(await add_words_to_set(user["id"], set_id, pool_ids))


@router.delete("/sets/{set_id}/words/{pool_id}")
async def sets_words_remove(set_id: int, pool_id: int, user=Depends(get_current_user)):
    """Убрать слово из набора (прогресс SRS не трогаем)."""
    return _bad(await remove_word_from_set(user["id"], set_id, pool_id))


@router.post("/sets/{set_id}/generate")
async def sets_generate(set_id: int, body: dict, user=Depends(get_current_user)):
    """Сгенерировать слова через ИИ по теме/уровню/количеству (0–20) и добавить в набор.
    Тема — ключ темы или свободный текст; level — A1..C2 (необяз.); lang — язык переводов."""
    mark_activity()
    words = await get_set_words(user["id"], set_id)
    if words is None:
        raise HTTPException(status_code=404, detail="Not found")
    body = body or {}
    count = max(0, min(20, int(body.get("count") or 0)))
    if count > 0:
        from autofill import generate_set_words   # ленивый импорт — избегаем циклов на старте
        pids = await generate_set_words(body.get("topic"), body.get("level"), count, body.get("lang") or "ru")
        if pids:
            await add_words_to_set(user["id"], set_id, pids)
        words = await get_set_words(user["id"], set_id)
    return {"words": words}


def _parse_data_url(s):
    """data:image/jpeg;base64,XXXX → (mime, b64). Принимаем и «голый» base64 без префикса."""
    s = s or ""
    if s.startswith("data:"):
        head, _, b64 = s.partition(",")
        mime = (head[5:].split(";")[0]) or "image/jpeg"
        return mime, b64
    return "image/jpeg", s


@router.post("/sets/{set_id}/ocr")
async def sets_ocr(set_id: int, body: dict, user=Depends(get_current_user)):
    """Шаг 1 импорта с фото: распознать норвежские слова на изображении (Gemini vision).
    Возвращает ТОЛЬКО список слов — пользователь правит/удаляет, затем шлёт на /import-words.
    body: {image: data-URL|base64, hint?: уточнение промта}."""
    mark_activity()
    if (await get_set_words(user["id"], set_id)) is None:
        raise HTTPException(status_code=404, detail="Not found")
    image = (body or {}).get("image")
    if not image:
        raise HTTPException(status_code=400, detail="No image")
    mime, b64 = _parse_data_url(image)
    from autofill import words_from_image   # ленивый импорт — избегаем циклов на старте
    return {"words": await words_from_image(b64, mime, (body or {}).get("hint") or "")}


@router.post("/sets/{set_id}/import-words")
async def sets_import_words(set_id: int, body: dict, user=Depends(get_current_user)):
    """Шаг 2 импорта с фото: отредактированный список слов → обогащение обычным генератором → в набор.
    body: {words: [str], lang?: язык переводов}."""
    mark_activity()
    if (await get_set_words(user["id"], set_id)) is None:
        raise HTTPException(status_code=404, detail="Not found")
    words = [w for w in ((body or {}).get("words") or []) if isinstance(w, str) and w.strip()]
    if not words:
        raise HTTPException(status_code=400, detail="No words")
    from autofill import words_from_list   # ленивый импорт — избегаем циклов на старте
    pids = await words_from_list(words, (body or {}).get("lang") or "ru")
    if pids:
        await add_words_to_set(user["id"], set_id, pids)
    return {"words": await get_set_words(user["id"], set_id), "added": len(pids)}


@router.post("/sets/{set_id}/reset")
async def sets_reset(set_id: int, user=Depends(get_current_user)):
    """Сбросить рампу выученных слов набора (до звукового задания, не до карточки)."""
    return _bad(await reset_set_ramp(user["id"], set_id))


@router.get("/sets/{set_id}/session")
async def sets_session(set_id: int, size: int = 20, lang: str = "ru", user=Depends(get_current_user)):
    """Дрилл «Учить набор»: сессия только по словам набора (рампа из общего SRS,
    без авто-добора из Базы и без лимита WIP/ворот — учим то, что выбрано)."""
    words = await get_set_words(user["id"], set_id)
    if words is None:
        raise HTTPException(status_code=404, detail="Not found")
    return await learning_session(user["id"], size=max(1, min(50, size)), lang=lang, set_id=set_id)
