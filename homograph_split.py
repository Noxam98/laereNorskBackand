"""LLM-классификация омонимов (сущ./глаг./прил.) в пуле и разбиение на per-pos записи.

Эвристика по русским окончаниям НЕ работает (быть/мочь короткие; «прочность»/-ость — ложняк),
поэтому части речи определяет LLM: каждый русский перевод → к своей части речи. Запись оставляет
значения СВОЕЙ части речи, остальные части речи → отдельные записи-омонимы (norwegian, pos).
dry_run=True — только план, БД не трогаем (после прошлого инцидента — сперва смотрим, потом применяем).
"""
import asyncio
import json
import logging
import time

import errors
import runtime
from llm import ask_json
from db.core import _conn, _release, normalize_word
from db.pool import get_or_create_pool, update_pool_translate, get_pool_by_id

_POS = ("noun", "verb", "adjective", "adverb")
_SYS = (
    "Ты лексикограф норвежского (bokmål). На входе норвежское слово и его русские переводы вперемешку. "
    "Определи часть речи КАЖДОГО русского перевода (по самому РУССКОМУ слову) и положи в корзину:\n"
    "- noun (существительное): предмет/понятие/лицо, «кто/что» — лайк, голос, сила, ошибка, поддержка, бонус;\n"
    "- verb (глагол): действие, русский ИНФИНИТИВ на -ть/-ти/-чь — нравиться, голосовать, поддерживать;\n"
    "- adjective (прилагательное): признак, «какой» — неправильный, сильный, гордый;\n"
    "- adverb (наречие): «как/насколько» — быстро, очень, особенно.\n"
    "Заимствования и вещи (лайк, бонус, фокус) — это noun, НЕ adjective. Ничего не выдумывай и не добавляй — "
    "только разложи ДАННЫЕ значения. Все значения одной части речи → все в одну корзину."
)
_STR_ARR = {"type": "array", "items": {"type": "string"}}
_SCHEMA = {
    "name": "homograph_split",
    "schema": {
        "type": "object",
        "properties": {"results": {"type": "array", "items": {
            "type": "object",
            "properties": {"word": {"type": "string"}, "noun": _STR_ARR, "verb": _STR_ARR,
                           "adjective": _STR_ARR, "adverb": _STR_ARR},
            "required": ["word"],
        }}},
        "required": ["results"],
    },
}


async def _candidates(limit):
    """Записи (сущ./глаг./прил.) с ≥2 рус. переводами — потенциальные омонимы. Частотные первыми."""
    db = await _conn()
    try:
        out = []
        async with db.execute(
            "SELECT id, norwegian, data, freq, COALESCE(pos,'') p FROM word_pool "
            "WHERE lower(COALESCE(pos,'')) IN ('noun','verb','adjective')") as cur:
            for r in await cur.fetchall():
                d = json.loads(r["data"]) if r["data"] else {}
                ru = [x.strip() for x in (d.get("translate", {}).get("ru") or []) if x and x.strip()]
                if len(ru) >= 2:
                    out.append({"id": r["id"], "no": r["norwegian"], "pos": r["p"], "ru": ru, "freq": r["freq"] or 0})
        out.sort(key=lambda c: -c["freq"])
        return out[:limit] if limit else out
    finally:
        await _release(db)


def _buckets(item):
    b = {}
    for k in _POS:
        v = [x.strip() for x in (item.get(k) or []) if isinstance(x, str) and x.strip()]
        if v:
            b[k] = v
    return b


async def _process(rows, apply, batch=15):
    """LLM-классификация rows по части речи → план разбиения; применить при apply=True.
    Делим запись ТОЛЬКО если LLM нашёл ≥2 части речи И у исходного pos есть значения."""
    by_key = {normalize_word(c["no"]): c for c in rows}
    plan = []
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        user = "Слова:\n" + "\n".join(f"- {c['no']}: {', '.join(c['ru'])}" for c in chunk)
        try:
            res = await ask_json(_SYS, user, _SCHEMA, purpose="autofill",
                                 label=f"омонимы ({len(chunk)})", temperature=0)
        except Exception as e:
            errors.report(e, "homograph")
            continue
        for item in (res or {}).get("results", []):
            c = by_key.get(normalize_word(item.get("word") or ""))
            if not c:
                continue
            b = _buckets(item)
            if c["pos"] not in b or len(b) < 2:
                continue   # одна часть речи (или у исходного pos нет значений) — не делим
            plan.append({"id": c["id"], "word": c["no"], "pos": c["pos"],
                         "keep_ru": b[c["pos"]], "others": {k: v for k, v in b.items() if k != c["pos"]}})
    if apply:
        for p in plan:
            row = await get_pool_by_id(p["id"])
            data = (row or {}).get("data") or {}
            if isinstance(data, str):
                data = json.loads(data)
            tr = dict((data or {}).get("translate", {}))
            tr["ru"] = p["keep_ru"]
            await update_pool_translate(p["id"], tr)   # оригинал: оставить значения своей части речи
            for opos, oru in p["others"].items():       # прочие части речи → отдельные записи-омонимы
                await get_or_create_pool(p["word"], {"word": p["word"], "part_of_speech": opos,
                                                     "translate": {"ru": oru}})
    return plan


async def split_homographs(dry_run=True, limit=None):
    """Топ-`limit` кандидатов (по частоте): план (dry) или применить."""
    return await _process(await _candidates(limit), apply=not dry_run)


_LOG_CAP = 120   # сколько последних разбиений держим в истории (homograph_log)


async def _save_progress(cands, done, plan):
    """Дешёвая сводка (homograph_stats) + дописать историю (homograph_log). cands/done уже посчитаны
    в батче — повторного скана пула НЕТ. Именно это читает админка (никакого скана на загрузке)."""
    from db import get_setting, set_setting
    total = len(cands)
    checked = sum(1 for c in cands if c["id"] in done)
    prev = await get_setting("homograph_stats")
    splits = (json.loads(prev).get("splits", 0) if prev else 0) + len(plan)
    await set_setting("homograph_stats", json.dumps({
        "checked": checked, "total": total, "remaining": total - checked,
        "splits": splits, "updated": int(time.time())}))
    if plan:
        raw = await get_setting("homograph_log")
        log = json.loads(raw) if raw else []
        ts = int(time.time())
        for p in plan:
            log.append({"word": p["word"], "pos": p["pos"], "keep": p["keep_ru"],
                        "others": p["others"], "ts": ts})
        await set_setting("homograph_log", json.dumps(log[-_LOG_CAP:], ensure_ascii=False))


async def homograph_batch(n=30):
    """Обработать СЛЕДУЮЩИЕ n непроверенных кандидатов (трекинг id в setting homograph_done), разбить
    подтверждённые омонимы, пометить пройденными + обновить сводку/историю (homograph_stats,
    homograph_log) для админки. Возвращает (взято, разбито, plan)."""
    from db import get_setting, set_setting
    raw = await get_setting("homograph_done")
    done = set(json.loads(raw)) if raw else set()
    cands = await _candidates(None)
    todo = [c for c in cands if c["id"] not in done][:n]
    if not todo:
        await _save_progress(cands, done, [])   # сводку всё равно освежим (total/remaining мог измениться)
        return 0, 0, []
    plan = await _process(todo, apply=True)
    done |= {c["id"] for c in todo}
    await set_setting("homograph_done", json.dumps(sorted(done)))
    await _save_progress(cands, done, plan)
    return len(todo), len(plan), plan


async def homograph_summary():
    """Готовая сводка для админки: ТОЛЬКО чтение settings — без скана пула, без LLM, без сплита."""
    from db import get_setting
    raw = await get_setting("homograph_stats")
    stats = json.loads(raw) if raw else {"checked": 0, "total": 0, "remaining": 0, "splits": 0, "updated": 0}
    rawlog = await get_setting("homograph_log")
    log = json.loads(rawlog) if rawlog else []
    stats["recent"] = list(reversed(log))   # новые сверху
    stats["paused"] = bool(runtime.PAUSED.get("homograph"))
    return stats


logger = logging.getLogger("learnnorsk")
HOMOGRAPH_BATCH = 20
HOMOGRAPH_SEC = 12   # пауза между батчами — пейсинг LLM/квоты (429 ротация ключей — внутри клиента)


async def homograph_loop():
    """Фоновый воркер: идёт по пулу пачками (homograph_batch), делит подтверждённые омонимы сущ./
    глаг./прил. на per-pos записи; пейсит под квоту. Пауза — runtime.PAUSED['homograph'] (админка).
    Когда все кандидаты пройдены (homograph_done) — спит долго (новые слова появляются редко)."""
    await asyncio.sleep(30)
    while True:
        if runtime.PAUSED.get("homograph"):
            await asyncio.sleep(30)
            continue
        try:
            seen, split, _ = await homograph_batch(HOMOGRAPH_BATCH)
        except Exception as e:
            errors.report(e, "homograph_loop")
            await asyncio.sleep(60)
            continue
        if seen == 0:
            await asyncio.sleep(600)
            continue
        if split:
            logger.info(f"homograph: +{split} разбито (из {seen})")
        await asyncio.sleep(HOMOGRAPH_SEC)
