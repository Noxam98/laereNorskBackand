"""Генерация и импорт слов в пул: по теме/уровню, по явному списку, OCR с фото; + восстановление «ё».
Зависит только от db/llm/task (не от autofill); реэкспортируется в autofill для воркеров и роутеров.
"""
import random
import errors
import llm
import runtime
from task import task
from llm import (
    ask_json, embed_texts, encode_emb, semantic_embed_text,
    WORDS_SCHEMA, normalize_word_item, apply_item_meta, LANG_NAMES, TOPIC_TAGS,
)
from db import (
    get_or_create_pool, get_pool_id, get_pool_words_by_names,
    set_pool_embedding, mark_sem_embed,
)


_YO_SYS = (
    "Ты — корректор русской буквы «ё». Дан список русских слов/фраз. Для КАЖДОГО верни написание с "
    "буквой «ё» там, где она орфографически НУЖНА (мёд, ёлка, всё, её, объём, полёт, актёр, ёж, тётя, "
    "лёд, своё, чёрный, идёт). Если «ё» не требуется — верни слово БЕЗ изменений. Меняй ТОЛЬКО е→ё там, "
    "где это правильно; больше НИЧЕГО не трогай (регистр, порядок слов, пунктуацию, прочие буквы сохраняй). "
    "Верни items: по объекту {src, fixed} на каждое входное слово."
)
_YO_SCHEMA = {"name": "yo", "schema": {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {"src": {"type": "string"}, "fixed": {"type": "string"}},
    "required": ["src", "fixed"]}}}, "required": ["items"]}}


async def restore_yo(words):
    """LLM-восстановление «ё» в списке русских строк. → {src: fixed}, только где реально изменилось
    и отличие РОВНО е→ё (защита от посторонних «правок» модели)."""
    if not words:
        return {}
    user = "Слова:\n" + "\n".join(f"- {w}" for w in words)
    try:
        data = await ask_json(_YO_SYS, user, _YO_SCHEMA, purpose="autofill", label=f"ё-фикс ({len(words)})")
    except Exception as e:
        errors.report(e, "restore_yo")
        return {}
    out = {}
    for it in (data.get("items", []) if isinstance(data, dict) else []):
        if not isinstance(it, dict):
            continue
        src, fixed = (it.get("src") or ""), (it.get("fixed") or "")
        if src and fixed and fixed != src and fixed.replace("ё", "е").replace("Ё", "Е") == src:
            out[src] = fixed
    return out
async def ai_game_words(lang, level, topic, count, on_phase=None):
    """AI-набор слов для онлайн-игры: 1 LLM-вызов под уровень/тему/язык (без «само-переводящихся»),
    прогон по пулу (переиспуск/создание), вектора новым (батч). Формы/tts добьются фоном.
    on_phase(p) — асинхронный колбэк прогресса ('indexing' перед эмбеддингами).
    Возвращает [{norwegian, translate, embedding}] для построения викторины."""
    lang_name = LANG_NAMES.get(lang, lang)
    topic_label = TOPIC_TAGS.get(topic, topic) if topic else None
    nonce = random.randint(1000, 99999)
    prompt = (
        f"Подбери {count + 3} распространённых, но РАЗНЫХ норвежских слов (bokmål) для игры-викторины. "
        + (f"Уровень CEFR — около {level}. " if level else "")
        + (f"Тема: {topic_label}. " if topic_label else "")
        + f"Игроки будут переводить их на язык: {lang_name}. "
        f"НЕ включай слова, чей перевод на {lang_name} совпадает или почти совпадает с самим норвежским "
        f"словом (интернационализмы/когнаты — они выдают ответ). Каждое слово однозначное и переводимое. "
        f"(вариант {nonce})"
    )
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                              purpose="autofill", label="AI-набор для игры")
    except Exception as e:
        errors.report(e, "ai_game_words")
        return []
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    names, new_emb = [], []   # имена для набора; (pid, текст) — новым на эмбеддинг
    for it in items:
        if not (isinstance(it, dict) and it.get("word") and not it.get("error")):
            continue
        tr = it.get("translate", {}) or {}
        target = (tr.get(lang) or [None])[0]
        if target and target.strip().lower() == it["word"].strip().lower():
            continue  # «само-переводящееся» — пропускаем
        existed = await get_pool_id(it["word"])
        data_item = normalize_word_item(it)
        pid = await get_or_create_pool(it["word"], data_item)
        if not pid:
            continue
        await apply_item_meta(pid, it)
        names.append(it["word"])
        if not existed:   # новое слово → нужен вектор
            new_emb.append((pid, semantic_embed_text(data_item) or it["word"]))
    await _embed_new(new_emb, on_phase)
    return await get_pool_words_by_names(names)


# модель для генерации набора: 3.5-flash (качество) → фолбэк 3.1-flash-lite (если 429/нет квоты)
SET_GEN_MODELS = ["gemini-3.5-flash", "gemini-3.1-flash-lite"]
VISION_MODELS = ["gemini-3.5-flash"]   # OCR/распознавание с фото — только vision-способная модель (не lite)
WORDS_ONLY_SCHEMA = {"name": "words", "schema": {"type": "object", "properties": {
    "words": {"type": "array", "items": {"type": "string"}}}, "required": ["words"]}}


async def _embed_new(new_emb, on_phase=None):
    """Посчитать и сохранить семантические эмбеддинги новым словам: new_emb = [(pool_id, текст)].
    on_phase — необязательный колбэк прогресса (для генерации слов игры)."""
    if not (new_emb and llm.embed_enabled() and not runtime.PAUSED["embed"]):
        return
    if on_phase:
        await on_phase("indexing")
    vecs = await embed_texts([t for _, t in new_emb])
    if vecs and len(vecs) == len(new_emb):
        for (pid, _), vec in zip(new_emb, vecs):
            await set_pool_embedding(pid, encode_emb(vec))
            await mark_sem_embed(pid)


async def _persist_word_items(items, n):
    """Положить items (формат WORDS_SCHEMA) в общий пул: перевод/мета/эмбеддинги. → pool_id без дублей."""
    pids, seen, new_emb = [], set(), []
    for it in items:
        if not (isinstance(it, dict) and it.get("word") and not it.get("error")):
            continue
        existed = await get_pool_id(it["word"])
        data_item = normalize_word_item(it)
        pid = await get_or_create_pool(it["word"], data_item)
        if not pid or pid in seen:
            continue
        await apply_item_meta(pid, it)
        seen.add(pid); pids.append(pid)
        if not existed:
            new_emb.append((pid, semantic_embed_text(data_item) or it["word"]))
        if len(pids) >= n:
            break
    await _embed_new(new_emb)   # вектора новым словам — чтобы сразу участвовали в подборе/похожих
    return pids


async def generate_set_words(topic, level, count, lang="ru"):
    """AI-набор слов для ЛИЧНОГО набора: тематическая генерация под уровень/количество (0–20).
    Модель — 3.5-flash с фолбэком на 3.1-flash-lite. Слова кладём в общий пул (обогащение фоном).
    Возвращает список pool_id добавленных/существующих слов (без дублей)."""
    n = max(1, min(20, int(count or 0)))
    lang_name = LANG_NAMES.get(lang, lang)
    topic_txt = (topic or "").strip()
    topic_label = TOPIC_TAGS.get(topic_txt, topic_txt) if topic_txt else None
    nonce = random.randint(1000, 99999)
    prompt = (
        f"Подбери РОВНО {n} распространённых, но РАЗНЫХ норвежских слов (bokmål) для изучения. "
        + (f"Уровень CEFR — около {level}. " if level else "")
        + (f"Тема: {topic_label}. " if topic_label else "Тема: общеупотребительная лексика. ")
        + f"Для каждого дай перевод на язык: {lang_name}. Каждое слово однозначное и переводимое. "
        f"(вариант {nonce})"
    )
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                              purpose="user", model=SET_GEN_MODELS, label="AI-набор слов")
    except Exception as e:
        errors.report(e, "generate_set_words")
        return []
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    return await _persist_word_items(items, n)


async def words_from_image(image_b64, mime="image/jpeg", hint="", limit=30):
    """OCR через Gemini vision: вытащить норвежские слова с изображения. Возвращает ТОЛЬКО список слов
    (без перевода) — дальше их обогащает обычный генератор. hint — необязательное уточнение от юзера."""
    sys = ("Du er en OCR-assistent for norskelever. Finn ALLE norske ord og korte uttrykk på bildet. "
           "Gi GRUNNFORM/oppslagsform når mulig, uten duplikater. Returner KUN ordene — ingen oversettelse, "
           "ingen forklaring, ingen tall eller rene symboler. Hopp over ord som ikke er norske.")
    extra = (hint or "").strip()
    user = [{"type": "text", "text": extra or "Hent de norske ordene fra dette bildet."},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}}]
    try:
        data = await ask_json(sys, user, WORDS_ONLY_SCHEMA, purpose="user", model=VISION_MODELS, label="OCR слов с фото")
    except Exception as e:
        errors.report(e, "words_from_image")
        return []
    raw = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    out, seen = [], set()
    for w in raw:
        w = w.strip() if isinstance(w, str) else ""
        k = w.lower()
        if w and k not in seen:
            seen.add(k); out.append(w)
        if len(out) >= limit:
            break
    return out


async def words_from_list(words, lang="ru"):
    """«Обычный генератор» для ЯВНОГО списка слов: обогащаем (перевод/часть речи/уровень) и кладём в пул.
    НЕ выдумывает новых слов — только то, что в списке (распознанное с фото). → список pool_id."""
    words = [w for w in (words or []) if isinstance(w, str) and w.strip()][:20]
    if not words:
        return []
    lang_name = LANG_NAMES.get(lang, lang)
    lst = "; ".join(words)
    prompt = (f"Вот ГОТОВЫЙ список норвежских слов (bokmål), распознанных с фото: {lst}. "
              f"Для КАЖДОГО слова из списка дай перевод на язык: {lang_name}, часть речи и уровень CEFR, "
              f"приведи к нормальной (словарной) форме. НЕ добавляй слов, которых нет в списке; "
              f"нераспознаваемое/не-норвежское — пропусти.")
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                              purpose="user", model=SET_GEN_MODELS, label="обогащение слов с фото")
    except Exception as e:
        errors.report(e, "words_from_list")
        return []
    items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
    return await _persist_word_items(items, len(words))
