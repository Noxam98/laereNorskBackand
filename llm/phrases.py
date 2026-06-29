"""Генерация устойчивых выражений (коллокации / фразовые глаголы / выражения) для рампы «фразы».

Фраза хранится как запись пула pos='phrase': норвежская фраза — в norwegian, а в data —
учебные данные (translate/example) и игровые под ключом `game` (game.distractors для игры
«порядок слов»). Дистракторы — ОДНОсловные кандидаты, которые фронт подмешивает к словам
фразы; игрок собирает точную последовательность токенов. persist (get_or_create_pool) дедупит
по (norwegian, pos), поэтому повтор не создаёт дубль."""
import re

from .client import ask_json
from .schemas import PHRASES_SCHEMA, CEFR_LEVELS

_SUBTYPES = ("collocation", "phrasal_verb", "expression")

_PHRASE_SYS = (
    "Ты эксперт норвежского (букмол) и методист. Генерируешь УСТОЙЧИВЫЕ выражения (коллокации, "
    "frasalverb, фиксированные выражения), реально частотные для уровня. Для каждого:\n"
    "- phrase: норвежская фраза в нижнем регистре, БЕЗ инфинитивной частицы 'å' в начале, без финальной точки;\n"
    "- translate: ru/ukr/en/pl/lt (1-2 варианта);\n- subtype; level; example (короткое предложение);\n"
    "- distractors: 2-3 ЛИШНИХ слова для игры «собери порядок слов». ЖЁСТКИЕ ПРАВИЛА:\n"
    "  • КАЖДЫЙ дистрактор — РОВНО ОДНО норвежское слово (без пробелов, без артикля);\n"
    "  • правдоподобная замена одному из слов фразы в той же грамматической роли (не тот глагол/предлог/сущ. из того же поля);\n"
    "  • дистрактор НЕ входит во фразу;\n"
    "  • дистрактор НЕ должен быть СИНОНИМОМ, дающим ту же фразу с тем же смыслом (результат с дистрактором должен быть ОШИБОЧНЫМ или иметь другой смысл).\n"
    "Строго по схеме."
)


def _norm_phrase(p):
    """Нормализовать фразу: нижний регистр, снять финальную точку и инфинитивное 'å '."""
    p = (p or "").strip().lower().rstrip(".")
    return re.sub(r"^å\s+", "", p).strip()


def clean_phrase_item(raw):
    """Сырой элемент LLM → item для persist_pool (как у слова: word/part_of_speech/translate/…),
    либо None если брак. Валидация: фраза ≥2 слов; дистракторы — только одно-словные, не входящие
    во фразу, без дублей, ≥2 шт.; перевод непустой."""
    if not isinstance(raw, dict):
        return None
    ph = _norm_phrase(raw.get("phrase", ""))
    toks = [t for t in ph.split() if t]
    if len(toks) < 2:                       # «фраза» из одного слова — не фраза
        return None
    tokset = set(toks)
    ds, seen = [], set()
    for d in (raw.get("distractors") or []):
        dd = (d or "").strip().lower()
        if not dd or " " in dd or dd in tokset or dd in seen:
            continue                         # только одно слово, не из фразы, без повторов
        seen.add(dd)
        ds.append(dd)
    if len(ds) < 2:
        return None
    tr = raw.get("translate") or {}
    translate = {"no": [ph]}                 # как normalize_word_item для слов: no = сама фраза
    for k in ("ru", "ukr", "en", "pl", "lt"):
        vals = [s.strip() for s in (tr.get(k) or []) if isinstance(s, str) and s.strip()]
        if vals:
            translate[k] = vals
    if len(translate) == 1:                  # кроме 'no' нет ни одного перевода — брак
        return None
    return {
        "word": ph,
        "part_of_speech": "phrase",
        "translate": translate,
        "level": raw.get("level") if raw.get("level") in CEFR_LEVELS else None,
        "subtype": raw.get("subtype") if raw.get("subtype") in _SUBTYPES else "collocation",
        "example": (raw.get("example") or "").strip() or None,
        "game": {"distractors": ds[:3]},
    }


def _zipf_to_cefr(z):
    """Грубый провизорный уровень дистрактора по частоте (zipf) — пока classify-луп не уточнит."""
    if z is None:
        return None
    if z >= 4.8: return "A1"
    if z >= 4.2: return "A2"
    if z >= 3.6: return "B1"
    if z >= 3.0: return "B2"
    return "C1"


async def pool_back_distractors(items):
    """Завести дистракторы фраз в Базу (если их там ещё нет) — с провизорным уровнем из частоты,
    чтобы фильтр узнаваемости в сессии (build_session) работал сразу; classify/translate-лупы
    дозаполнят позже. items — список phrase-items (с item['game']['distractors']) или строк.
    Возвращает число заведённых. ВЫЗЫВАТЬ после генерации/persist новых фраз."""
    from db import get_or_create_pool, get_pool_id, set_pool_meta
    from db.core import _conn, _release
    ds = set()
    for it in items:
        if isinstance(it, str):
            ds.add(it.strip().lower())
        elif isinstance(it, dict):
            for d in ((it.get("game") or {}).get("distractors") or []):
                ds.add((d or "").strip().lower())
    ds.discard("")
    added = 0
    for d in sorted(ds):
        if await get_pool_id(d):                 # уже в Базе (любой pos) — пропускаем
            continue
        db = await _conn()
        try:
            async with db.execute("SELECT zipf FROM nb_lexicon WHERE word = ? LIMIT 1", (d,)) as c:
                r = await c.fetchone()
                z = r["zipf"] if r else None
        finally:
            await _release(db)
        pid = await get_or_create_pool(d, {"word": d, "part_of_speech": "", "translate": {"no": [d]}}, approved=1)
        if pid:
            lvl = _zipf_to_cefr(z)
            if lvl:
                await set_pool_meta(pid, level=lvl)
            added += 1
    return added


async def generate_phrases(level, n=8, model=None):
    """~n устойчивых выражений уровня `level` → список валидных items (после отбраковки может быть
    < n). Дедуп против пула/повторов прогона — на стороне вызывающего."""
    user = (f"Сгенерируй {n} частотных норвежских устойчивых выражений уровня {level}. "
            f"Смесь коллокаций, фразовых глаголов и выражений.")
    res = await ask_json(_PHRASE_SYS, user, PHRASES_SCHEMA, purpose="user", label=f"phrases {level}", model=model)
    raw = (res or {}).get("phrases", []) if isinstance(res, dict) else []
    out = []
    for r in raw:
        it = clean_phrase_item(r)
        if it:
            out.append(it)
    return out
