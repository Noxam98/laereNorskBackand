"""Генерация и импорт слов в пул: по теме/уровню, по явному списку, OCR с фото; + восстановление «ё».
Зависит только от db/llm/task (не от autofill); реэкспортируется в autofill для воркеров и роутеров.
"""
import asyncio
import json
import os
import random
import re
import errors
import llm
import runtime
from task import task
from llm import (
    ask_json, embed_texts, encode_emb, semantic_embed_text,
    WORDS_SCHEMA, normalize_word_item, apply_item_meta, LANG_NAMES, TOPIC_TAGS,
)
from db import (
    get_or_create_pool, get_pool_id, get_pool_by_id, get_pool_words_by_names,
    get_import_pool_candidates,
    normalize_word, set_pool_embedding, mark_sem_embed, set_pool_forms, set_pool_meta,
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
async def ai_game_words(lang, level, topic, count, on_phase=None, created_by=None, approved=1):
    """AI-набор слов для онлайн-игры: 1 LLM-вызов под уровень/тему/язык (без «само-переводящихся»),
    прогон по пулу (переиспуск/создание), вектора новым (батч). Формы/tts добьются фоном.
    on_phase(p) — асинхронный колбэк прогресса ('indexing' перед эмбеддингами).
    created_by/approved — модерация: пользовательский маршрут (/games/ai_words) кладёт слова в
    личное расширение автора (approved=0), онлайн-хост оставляет дефолт (approved=1).
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
        pid = await get_or_create_pool(it["word"], data_item, created_by=created_by, approved=approved)
        if not pid:
            continue
        await apply_item_meta(pid, it)
        names.append(it["word"])
        if not existed:   # новое слово → нужен вектор
            new_emb.append((pid, semantic_embed_text(data_item) or it["word"]))
    await _embed_new(new_emb, on_phase)
    return await get_pool_words_by_names(names)


# Генерация набора: сильная 3.6 Flash → дешёвый 3.5 Flash-Lite при 429/перегрузке.
SET_GEN_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
VISION_MODELS = ["gemini-3.6-flash"]   # OCR важнее точность; Lite остаётся для массовых текстовых задач
WORDS_ONLY_SCHEMA = {"name": "words", "schema": {"type": "object", "properties": {
    "words": {"type": "array", "items": {"type": "string"}}}, "required": ["words"]}}
TEXT_ITEMS_SCHEMA = {"name": "import_items", "schema": {"type": "object", "properties": {
    "items": {"type": "array", "items": {"type": "object", "properties": {
        "word": {"type": "string"}, "translation": {"type": "string"},
    }, "required": ["word", "translation"]}},
}, "required": ["items"]}}


class ImportProviderError(RuntimeError):
    """Провайдер не обработал импорт: роут должен отличить это от честного пустого результата."""


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


async def _persist_word_items(items, n, created_by=None, approved=1, detailed=False):
    """Положить items (формат WORDS_SCHEMA) в общий пул: перевод/мета/эмбеддинги. → pool_id без дублей.
    created_by/approved пробрасываются в get_or_create_pool: пользовательские импорты/генерация кладут
    слова в личное расширение автора (approved=0), фоновые вызовы оставляют дефолт (approved=1)."""
    pids, details, seen, new_emb = [], [], set(), []
    for it in items:
        if not (isinstance(it, dict) and it.get("word") and not it.get("error")):
            continue
        data_item = normalize_word_item(it)
        pid, created = await get_or_create_pool(
            it["word"], data_item, created_by=created_by, approved=approved, return_created=True,
        )
        if not pid or pid in seen:
            continue
        await apply_item_meta(pid, it)
        seen.add(pid); pids.append(pid)
        details.append({"pid": pid, "word": it["word"], "created": created})
        if created:
            new_emb.append((pid, semantic_embed_text(data_item) or it["word"]))
        if len(pids) >= n:
            break
    await _embed_new(new_emb)   # вектора новым словам — чтобы сразу участвовали в подборе/похожих
    return details if detailed else pids


async def generate_set_words(topic, level, count, lang="ru", created_by=None):
    """AI-набор слов для ЛИЧНОГО набора: тематическая генерация под уровень/количество (0–20).
    Модель — 3.5-flash с фолбэком на 3.1-flash-lite. Слова кладём в личное расширение автора
    (approved=0, created_by=user) — модерация до попадания в общую Базу; обогащение фоном.
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
    return await _persist_word_items(items, n, created_by=created_by, approved=0)


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
        raise ImportProviderError("ocr_failed") from e
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


_PAIR_RE = re.compile(r"^\s*(?:[-*•]\s*)?(.{1,80}?)\s+(?:—|–|=)\s+(.{1,240}?)\s*$")


def _explicit_text_items(text, limit=50):
    """Без LLM разобрать чистый список «norsk — перевод», сохранив пользовательский перевод."""
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return []
    items = []
    for line in lines:
        match = _PAIR_RE.match(line)
        if not match:
            return None
        items.append({"word": match.group(1).strip(), "translation": match.group(2).strip()})
        if len(items) >= limit:
            break
    return items


def _clean_text_items(raw, limit=50):
    out, seen = [], set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        word = (item.get("word") or "").strip()
        translation = (item.get("translation") or "").strip()
        key = word.lower()
        if not word or key in seen:
            continue
        seen.add(key)
        out.append({"word": word, "translation": translation})
        if len(out) >= limit:
            break
    return out


async def items_from_text(text, hint="", limit=50):
    """Извлечь структурированные слова, не теряя явно указанный пользователем перевод."""
    text = (text or "").strip()
    if not text:
        return []
    explicit = _explicit_text_items(text, limit)
    if explicit is not None:
        return _clean_text_items(explicit, limit)
    sys = (
        "Du rydder opp i en fritekstliste med gloser for en norskelever. "
        "Returner hvert element som {word, translation}. word skal være norsk bokmål i grunnform. "
        "For «X — Y», «X – Y» eller «X = Y» skal translation være Y NØYAKTIG som brukeren skrev den. "
        "For norske ord uten oversettelse skal translation være en tom streng. Hvis listen bare er på "
        "et annet språk, oversett ordet til norsk og behold originalen som translation. Ignorer navn, "
        "tidsstempler, nummerering, lenker, rene symboler og duplikater."
    )
    extra = (hint or "").strip()
    user = (f"{extra}\n\n" if extra else "") + f"Tekst:\n{text}"
    try:
        data = await ask_json(sys, user, TEXT_ITEMS_SCHEMA, purpose="user",
                              model=SET_GEN_MODELS, label="разбор слов из текста")
    except Exception as e:
        errors.report(e, "items_from_text")
        raise ImportProviderError("parse_failed") from e
    raw = data.get("items", []) if isinstance(data, dict) else []
    return _clean_text_items(raw, limit)


async def words_from_text(text, hint="", limit=50):
    """Импорт слов из ПРОИЗВОЛЬНОГО текста (фото-аналог, но текстом): чат-логи с метками времени и
    именами, списки через запятую/перенос, строки «norsk — перевод» (берём норвежскую сторону) и даже
    список на другом языке без перевода (тогда переводим на норвежский). Возвращает ТОЛЬКО список
    норвежских слов (без перевода) — дальше их обогащает words_from_list. hint — необязательное уточнение."""
    return [item["word"] for item in await items_from_text(text, hint, limit)]


async def words_from_list(words, lang="ru", limit=50, created_by=None):
    """«Обычный генератор» для ЯВНОГО списка слов: обогащаем (перевод/часть речи/уровень) и кладём в пул.
    НЕ выдумывает новых слов — только то, что в списке. Длинные списки (импорт текстом) обрабатываем
    пачками по 20 — размер, под который настроены модель/схема. Пользовательский импорт → личное
    расширение автора (approved=0, created_by=user), модерация до общей Базы. → список pool_id (без дублей)."""
    report = await import_words_from_list(words, lang, limit, created_by)
    return report["pool_ids"]


def _clean_import_items(items, limit=50):
    out, by_key = [], {}
    for value in items or []:
        if isinstance(value, str):
            word, translation = value.strip(), ""
        elif isinstance(value, dict):
            word = (value.get("word") or "").strip()
            translation = (value.get("translation") or value.get("provided_translation") or "").strip()
        else:
            continue
        if not word or len(word) > 80:
            continue
        key = word.lower()
        if key in by_key:
            if translation and not by_key[key]["translation"]:
                by_key[key]["translation"] = translation
            continue
        item = {"word": word, "translation": translation}
        by_key[key] = item
        out.append(item)
        if len(out) >= limit:
            break
    return out


_LEVELS = None


def _bank_level(word, pos):
    """Уровень из опубликованного словника; грузим файл лениво только при новом слове из банка."""
    global _LEVELS
    if _LEVELS is None:
        try:
            path = os.path.join(os.path.dirname(__file__), "data", "levels-v1.json")
            with open(path, encoding="utf-8") as src:
                _LEVELS = json.load(src).get("levels", {})
        except Exception:
            _LEVELS = {}
    return _LEVELS.get(f"{word}|{pos}")


def _bank_candidate(word):
    """Однозначное соответствие Ordbank: лемма либо словоформа → лемма.

    Несколько частей речи/лемм не угадываем — такой случай остаётся AI, где перевод и контекст
    помогают выбрать нужный омоним.
    """
    from db import ordbank

    key = normalize_word(word)
    direct = [(key, pos, ordbank.lookup(key, pos)) for pos in ("noun", "verb", "adjective")]
    candidates = [(lemma, pos, forms) for lemma, pos, forms in direct if forms]
    if not candidates:
        candidates = [
            (lemma, pos, ordbank.lookup(lemma, pos))
            for lemma, pos in ordbank.exact_form(key)
        ]
        candidates = [(lemma, pos, forms) for lemma, pos, forms in candidates if forms]
    unique = {(lemma, pos): forms for lemma, pos, forms in candidates}
    if len(unique) != 1:
        return None
    (lemma, pos), forms = next(iter(unique.items()))
    return {"input_key": key, "word": lemma, "pos": pos, "forms": forms}


async def resolve_bank_words(items, lang="ru", created_by=None, concurrency=4):
    """Разрешить список через штатную цепочку Ordbank/Lexin до обращения к AI.

    Возвращает только однозначно разрешённые элементы. Lexin ограничен по параллельности, затем
    новые записи и эмбеддинги сохраняются одной пачкой через тот же примитив, что AI-генерация.
    """
    entries = _clean_import_items(items, len(items or []) or 1)
    gate = asyncio.Semaphore(max(1, min(int(concurrency or 1), 6)))

    async def prepare(entry):
        bank = _bank_candidate(entry["word"])
        if not bank:
            return None
        pid = await get_pool_id(bank["word"], bank["pos"])
        if pid:
            current = await get_pool_by_id(pid, user_id=created_by)
            if current:
                return {
                    **bank, "entry": entry, "pid": pid, "created": False, "source": "pool",
                    "translate": current.get("translate", {}) or {},
                }
        provided = entry["translation"]
        if provided:
            translations, source = {lang: [provided]}, "ordbank"
        else:
            import lexin as lexin_live
            async with gate:
                translations = await lexin_live.lookup(bank["word"], bank["pos"])
            source = "lexin"
        if not translations:
            return None
        translations = {**translations, "no": [bank["word"]]}
        if provided:
            translations[lang] = [provided]
        return {
            **bank, "entry": entry, "pid": None, "created": None, "source": source,
            "translate": translations,
            "pool_item": {
                "word": bank["word"], "part_of_speech": bank["pos"], "translate": translations,
            },
        }

    prepared = [item for item in await asyncio.gather(*(prepare(entry) for entry in entries)) if item]
    new_items = [item["pool_item"] for item in prepared if item.get("pool_item")]
    details = await _persist_word_items(
        new_items, len(new_items), created_by=created_by, approved=0, detailed=True,
    ) if new_items else []
    created_by_pid = {item["pid"]: bool(item["created"]) for item in details}

    resolved = []
    for item in prepared:
        pid = item["pid"] or await get_pool_id(item["word"], item["pos"])
        current = await get_pool_by_id(pid, user_id=created_by) if pid else None
        if not current:
            continue
        if item.get("pool_item"):
            await set_pool_forms(pid, item["forms"])
            level = _bank_level(item["word"], item["pos"])
            if level:
                await set_pool_meta(pid, level=level)
        resolved.append({
            "input_key": item["input_key"],
            "input": item["entry"]["word"],
            "word": item["word"],
            "pid": pid,
            "created": created_by_pid.get(pid, False),
            "source": item["source"],
            "via_form": item["input_key"] if item["input_key"] != item["word"] else None,
            "translate": current.get("translate", {}) or item["translate"],
        })
    return resolved


async def import_words_from_list(words, lang="ru", limit=50, created_by=None):
    """Переиспользовать пул и Ordbank/Lexin до LLM, новые/омонимы обогатить пачками.

    Возвращает детальный отчёт: роут на его основе показывает частичный успех, а не закрывает
    импорт как полностью успешный после сбоя одной пачки.
    """
    entries = _clean_import_items(words, limit)
    if not entries:
        return {
            "pool_ids": [], "requested": 0, "reused": 0, "created": 0,
            "failed": [], "overrides": {},
        }
    candidates = await get_import_pool_candidates(created_by, [item["word"] for item in entries])
    pids, seen, pending, failed = [], set(), [], []
    overrides = {}
    reused = created = 0
    for item in entries:
        matches = candidates.get(item["word"].lower(), [])
        if len(matches) == 1:
            pid = matches[0]["id"]
            if pid not in seen:
                seen.add(pid); pids.append(pid); reused += 1
            if item["translation"]:
                overrides[pid] = {
                    "translate": {
                        **matches[0]["translate"],
                        lang: [item["translation"]],
                    },
                }
        else:
            pending.append(item)

    bank_resolved = {
        item["input_key"]: item
        for item in await resolve_bank_words(pending, lang=lang, created_by=created_by)
    }
    unresolved = []
    for item in pending:
        resolved = bank_resolved.get(normalize_word(item["word"]))
        if not resolved:
            unresolved.append(item)
            continue
        pid = resolved["pid"]
        if pid not in seen:
            seen.add(pid); pids.append(pid)
            if resolved["created"]:
                created += 1
            else:
                reused += 1
        if item["translation"] and not resolved["created"]:
            overrides[pid] = {
                "translate": {
                    **resolved["translate"],
                    lang: [item["translation"]],
                },
            }
    pending = unresolved

    lang_name = LANG_NAMES.get(lang, lang)
    for i in range(0, len(pending), 20):
        chunk = pending[i:i + 20]
        lines = "\n".join(
            f"- {item['word']}" + (f" — {item['translation']}" if item["translation"] else "")
            for item in chunk
        )
        prompt = (f"Вот ГОТОВЫЙ список норвежских слов (bokmål):\n{lines}\n"
                  f"Для КАЖДОГО слова из списка дай перевод на язык: {lang_name}, часть речи и уровень CEFR, "
                  f"приведи к нормальной (словарной) форме. НЕ добавляй слов, которых нет в списке; "
                  f"нераспознаваемое/не-норвежское — пропусти. Если после тире уже дан перевод, "
                  f"сохрани его без перефразирования.")
        try:
            data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                                  purpose="user", model=SET_GEN_MODELS, label="обогащение списка слов")
        except Exception as e:
            errors.report(e, "words_from_list")
            failed.extend({
                "word": item["word"], "translation": item["translation"], "reason": "provider_failed",
            } for item in chunk)
            continue
        items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        translations = {item["word"].lower(): item["translation"] for item in chunk if item["translation"]}
        for item in items:
            if not isinstance(item, dict):
                continue
            provided = translations.get((item.get("word") or "").strip().lower())
            if provided:
                item.setdefault("translate", {})[lang] = [provided]
        details = await _persist_word_items(items, len(chunk), created_by=created_by,
                                            approved=0, detailed=True)
        for detail in details:
            pid = detail["pid"]
            if pid not in seen:
                seen.add(pid); pids.append(pid)
                created += int(detail["created"])
            provided = translations.get(detail["word"].strip().lower())
            if provided and not detail["created"]:
                known = next(
                    (candidate for group in candidates.values() for candidate in group
                     if candidate["id"] == pid),
                    None,
                )
                overrides[pid] = {
                    "translate": {
                        **((known or {}).get("translate", {})),
                        lang: [provided],
                    },
                }
        if len(details) < len(chunk):
            returned = {(item.get("word") or "").strip().lower() for item in items if isinstance(item, dict)}
            missing = [item for item in chunk if item["word"].lower() not in returned]
            failed.extend({
                "word": item["word"], "translation": item["translation"], "reason": "not_recognized",
            } for item in missing)
    return {
        "pool_ids": pids,
        "requested": len(entries),
        "reused": reused,
        "created": created,
        "failed": failed,
        "overrides": overrides,
    }
