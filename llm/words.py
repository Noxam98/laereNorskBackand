"""Доменные операции над словами: генерация (по запросу/из кэша), уточнение переводов,
нормализация, проставление уровня/тем, сохранение в пул. Используют key-free примитивы
client.ask_json и embeddings.ensure_embeddings."""
from fastapi import HTTPException
import errors
from db import get_cached_query, cache_query, get_or_create_pool, set_pool_meta
from task import task
from .client import ask_json
from .quota import text_enabled
from .schemas import WORDS_SCHEMA, REFINE_SCHEMA, TOPIC_KEYS, CEFR_LEVELS, LANG_NAMES
from .embeddings import ensure_embeddings


def normalize_word_item(item):
    if not isinstance(item, dict) or item.get("error"):
        return item
    w = item.get("word")
    if w:
        tr = item.setdefault("translate", {})
        if not tr.get("no"):
            tr["no"] = [w]
    # разбор составного слова от LLM: валидируем — части ДОЛЖНЫ буква-в-букву складываться в слово
    # (иначе разбор неверный → дропаем, чтобы не хранить мусор; для слов из банка используется
    # авторитетный ordbank.compound, это только фолбэк для нюордов вне банка).
    c = item.get("compound")
    if isinstance(c, dict) and w:
        fl = (c.get("forledd") or "").strip().lower()
        fu = (c.get("fuge") or "").strip().lower()
        el = (c.get("etterledd") or "").strip().lower()
        if fl and el and (fl + fu + el) == w.strip().lower():
            item["compound"] = {"forledd": fl, "fuge": fu, "etterledd": el}
        else:
            item.pop("compound", None)
    elif "compound" in item:
        item.pop("compound", None)
    return item


async def apply_item_meta(pid, item):
    """Проставить уровень/темы из сгенерированного слова, если пришли валидными."""
    level = item.get("level") if item.get("level") in CEFR_LEVELS else None
    topics = [t for t in (item.get("topics") or []) if t in TOPIC_KEYS]
    if level or topics:
        await set_pool_meta(pid, level=level, topics=topics or None)


async def refine_translations(items, lang, model=None):
    """items: [{"word": no, "current": [..]}]. Вернуть {word_lower: [переводы]} —
    точные, различимые между собой переводы на язык `lang`."""
    lang_name = LANG_NAMES.get(lang, lang)
    lines = "\n".join(f"- {it['word']}: {', '.join(it.get('current') or []) or '—'}" for it in items)
    system = (
        f"Ты эксперт-лексикограф норвежского языка. Пользователь выделил группу слов, "
        f"у которых перевод на {lang_name} получился одинаковым или слишком общим. "
        f"Для КАЖДОГО слова дай точный перевод на {lang_name}, подобранный так, чтобы слова "
        f"в группе были различимы между собой (без идентичных переводов). 1-3 самых точных "
        f"варианта на слово, сохраняя часть речи. Отвечай строго по схеме."
    )
    user = f"Слова (норвежское: текущий перевод на {lang_name}):\n{lines}"
    res = await ask_json(system, user, REFINE_SCHEMA, purpose="user", label="уточнение переводов", model=model)
    out = {}
    if isinstance(res, dict):
        for r in res.get("results", []):
            w = (r.get("word") or "").strip().lower()
            tr = [t.strip() for t in (r.get("translate") or []) if isinstance(t, str) and t.strip()]
            if w and tr:
                out[w] = tr
    return out


async def generate_words(prompt, model=None):
    """Возвращает нормализованный список слов (из кэша или AI). model — необязательный
    override модели от пользователя; ключи/квота/учёт — внутри ask_json."""
    cached = await get_cached_query(prompt)
    if cached is not None:
        return cached, True
    if not text_enabled():
        raise HTTPException(status_code=503, detail="LLM is not configured (set LLM_API_KEY)")
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                              purpose="user", label="генерация слов", model=model)
    except Exception as e:
        info = errors.report(e, "generate_words")
        raise HTTPException(status_code=info.http_status, detail=info.user_detail)
    if data is None:
        raise HTTPException(status_code=500, detail="No JSON found in the response")
    # гарантированный формат: { words: [...] }
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    normalized = [normalize_word_item(i) for i in items]
    await cache_query(prompt, normalized)
    return normalized, False


async def persist_pool(normalized, created_by=None, approved=1):
    """Сохранить слова в пул + посчитать эмбеддинги ПАЧКОЙ. created_by/approved — для модерации:
    при генерации юзером (created_by=user, approved=0) новые слова идут в его личное расширение."""
    pairs = []
    for item in normalized:
        if isinstance(item, dict) and not item.get("error") and item.get("word"):
            pid = await get_or_create_pool(item["word"], item, created_by=created_by, approved=approved)
            if pid:
                await apply_item_meta(pid, item)
                pairs.append((pid, item))
    await ensure_embeddings(pairs)  # эмбеддинги всех новых слов одним запросом
