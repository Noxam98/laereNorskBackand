import os
import random
import asyncio
from datetime import datetime
from config import logger
import errors
from activity import seconds_idle
from db import (
    get_pool_id, get_pool_by_id, set_pool_embedding, get_pool_tts, set_pool_tts,
    get_or_create_pool, set_pool_meta, set_pool_description,
    pool_missing_embedding, pool_missing_tts, pool_missing_meta, pool_missing_description,
    translate_pending, mark_translate_done, update_pool_translate, normalize_word,
    sem_embed_pending, mark_sem_embed,
    get_pool_sample, get_pool_letter, missing_embedding_data, count_missing_embedding,
)
from llm import (
    LLM_API_KEY, LLM_API_KEYS, EMBED_API_KEYS, LLM_MODEL, EMBED_MODEL,
    embed_text, embed_texts, encode_emb, ask_json, WORDS_SCHEMA,
    CLASSIFY_SCHEMA, DESCRIBE_BATCH_SCHEMA, TRANSLATE_BATCH_SCHEMA, TOPIC_TAGS, TOPIC_KEYS, CEFR_LEVELS,
    normalize_word_item, apply_item_meta, semantic_embed_text,
    pick_key_model, incr_text_usage, incr_emb_usage,
)
from tts import synth_tts, _tts_lock
from task import task

# --- Авто-заполнение общего пула в рамках суточного бюджета ("свободная" квота) ---
AUTOFILL_ENABLED = os.getenv("AUTOFILL_ENABLED", "false").lower() == "true"

# Рантайм-флаги, управляемые из Telegram-бота (без редеплоя):
#   autofill — генерация новых слов (autofill_loop);
#   paused   — глобальная пауза ВСЕХ фоновых очередей (describe/translate/reembed/autofill).
RUNTIME = {"autofill": AUTOFILL_ENABLED, "paused": False}


def _bg_blocked(check_autofill=False):
    """Должна ли фоновая итерация пропуститься (пауза / выключенный autofill)."""
    if RUNTIME["paused"]:
        return True
    return check_autofill and not RUNTIME["autofill"]
AUTOFILL_DAILY_BUDGET = int(os.getenv("AUTOFILL_DAILY_BUDGET", "150"))
AUTOFILL_INTERVAL_SEC = int(os.getenv("AUTOFILL_INTERVAL_SEC", "150"))  # одно слово целиком раз в N сек (день)
AUTOFILL_BATCH = int(os.getenv("AUTOFILL_BATCH", "1"))
AUTOFILL_IDLE_SEC = int(os.getenv("AUTOFILL_IDLE_SEC", "300"))  # фон только после N сек простоя
# Ночной режим — агрессивнее (когда никто не пользуется)
AUTOFILL_NIGHT_INTERVAL_SEC = int(os.getenv("AUTOFILL_NIGHT_INTERVAL_SEC", "6"))   # ~10 операций/мин
AUTOFILL_NIGHT_BUDGET = int(os.getenv("AUTOFILL_NIGHT_BUDGET", "200"))             # потолок за день (оставляет запас квоты на день)
AUTOFILL_NIGHT_START = int(os.getenv("AUTOFILL_NIGHT_START", "2"))    # локальный час начала ночи
AUTOFILL_NIGHT_END = int(os.getenv("AUTOFILL_NIGHT_END", "7"))        # локальный час конца ночи
AUTOFILL_TZ_OFFSET = int(os.getenv("AUTOFILL_TZ_OFFSET", "2"))        # сдвиг от UTC (Норвегия летом = +2)
AUTOFILL_AVOID_SAMPLE = int(os.getenv("AUTOFILL_AVOID_SAMPLE", "60"))  # фикс-размер списка исключений в промпте
CLASSIFY_BATCH = int(os.getenv("CLASSIFY_BATCH", "50"))  # слов на один LLM-вызов классификации
# Ротация моделей по суточным лимитам Gemini free-tier (у каждой модели — свой RPD).
# Формат env: "model1:rpd1,model2:rpd2". Кончилась квота первой → берём следующую.
# Потолки берём чуть ниже реальных, оставляя запас живым запросам пользователей днём.
#   Текст: 3.1 Flash Lite=500; Gemma 4 26B/31B=1500 каждая; мелкие flash=20.
#   Эмбеддинги: Embedding 1=1000, Embedding 2=1000.
def _parse_models(env, default_model, default_budget):
    raw = (os.getenv(env, "") or "").strip()
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        m, _, b = part.rpartition(":")
        try:
            out.append((m.strip(), int(b)))
        except ValueError:
            out.append((part, default_budget))
    return out or [(default_model, default_budget)]

TEXT_MODELS = _parse_models("AUTOFILL_TEXT_MODELS", LLM_MODEL, int(os.getenv("TEXT_DAILY_BUDGET", "400")))
EMBED_MODELS = _parse_models("AUTOFILL_EMBED_MODELS", EMBED_MODEL, int(os.getenv("EMBED_DAILY_BUDGET", "800")))


async def _pick_text():
    """(ключ, модель) для текстового фонового запроса. (None, None) — всё исчерпано."""
    return await pick_key_model(TEXT_MODELS, LLM_API_KEYS, "text")


async def _pick_emb():
    """(ключ, модель) для эмбеддинга. (None, None) — всё исчерпано."""
    return await pick_key_model(EMBED_MODELS, EMBED_API_KEYS, "emb")


def _is_night():
    h = (datetime.utcnow().hour + AUTOFILL_TZ_OFFSET) % 24
    s, e = AUTOFILL_NIGHT_START, AUTOFILL_NIGHT_END
    return (s <= h < e) if s <= e else (h >= s or h < e)


AUTOFILL_TOPICS = [
    "семья и родственники", "еда и блюда", "напитки", "фрукты и овощи", "дом и жильё",
    "мебель", "кухня и посуда", "одежда и обувь", "аксессуары и украшения", "тело человека",
    "здоровье и болезни", "медицина и аптека", "гигиена и косметика", "эмоции и чувства",
    "черты характера", "отношения и любовь", "дружба и общение", "город и улицы",
    "здания и места", "транспорт и машины", "дорога и движение", "путешествия и туризм",
    "гостиница и отдых", "природа и ландшафт", "погода и климат", "времена года",
    "животные", "птицы", "рыбы и морские", "насекомые", "растения и деревья", "цветы",
    "сад и огород", "ферма и село", "море и пляж", "горы и лес", "небо и космос",
    "цвета и оттенки", "числа и счёт", "время и даты", "дни и месяцы", "работа и офис",
    "профессии", "школа и учёба", "наука", "технологии и компьютеры", "интернет и связь",
    "телефон и гаджеты", "деньги и финансы", "покупки и магазин", "рынок и продукты",
    "ресторан и кафе", "приготовление еды", "вкус и запах", "спорт и фитнес",
    "хобби и досуг", "музыка и инструменты", "искусство", "книги и литература",
    "кино и театр", "игры", "праздники и традиции", "религия", "политика и общество",
    "закон и право", "армия и оружие", "география и страны", "языки и народы",
    "инструменты и ремонт", "строительство и материалы", "энергия и экология",
    "виды транспорта", "безопасность и чрезвычайные", "детство и дети", "возраст и жизнь",
    "свадьба и события", "органы чувств", "ум и память", "сны и воображение",
    "качества и свойства", "размеры и формы", "материалы и вещества", "звуки и шумы",
    "свет и тьма", "движение и направление", "глаголы речи", "глаголы действия",
    "повседневные дела", "приветствия и вежливость", "вопросы и сомнения",
    "количество и меры", "абстрактные понятия", "бизнес и торговля", "экономика",
]

# Буквы норвежского алфавита с весами по частоте слов-начал (частые буквы выпадают чаще).
_LETTER_WEIGHTS = {
    "s": 12, "f": 9, "b": 9, "k": 9, "m": 8, "t": 8, "h": 7, "v": 7, "d": 7, "a": 6,
    "p": 6, "l": 6, "g": 5, "r": 5, "o": 4, "e": 4, "i": 4, "n": 4, "u": 3, "j": 3,
    "å": 3, "y": 2, "ø": 2, "æ": 1, "c": 1,
}
AUTOFILL_LETTERS = list(_LETTER_WEIGHTS.keys())
AUTOFILL_LETTER_W = list(_LETTER_WEIGHTS.values())


async def complete_word(w, emb_model=None, emb_key=None):
    """Доделать слово: эмбеддинг (если задана модель emb_model с бюджетом) + озвучка (всегда, edge бесплатный)."""
    pid = await get_pool_id(w)
    if emb_model and pid:
        p = await get_pool_by_id(pid)
        if p and not p.get("embedding"):
            vec = await embed_text(semantic_embed_text(p["data"]) or w, model=emb_model, api_key=emb_key)
            if vec:
                await set_pool_embedding(pid, encode_emb(vec))
                await mark_sem_embed(pid)
    if not await get_pool_tts(w):
        async with _tts_lock:
            try:
                mp3 = await synth_tts(w)
            except Exception as e:
                mp3 = None
                logger.warning(f"complete_word tts '{w}': {e}")
            if mp3:
                await set_pool_tts(w, mp3)


_CLASSIFY_SYS = (
    "Ты — лингвист-методист по норвежскому (bokmål). Для каждого слова определи: "
    "уровень CEFR (A1, A2, B1, B2, C1 или C2 — по частотности и сложности) и 1-3 темы "
    "СТРОГО из этого списка ключей:\n"
    + "; ".join(f"{k} ({v})" for k, v in TOPIC_TAGS.items())
    + ".\nВерни массив results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)


async def classify_batch(items, model=None, api_key=None):
    """Пакетная классификация (≤CLASSIFY_BATCH): уровень + темы за один LLM-вызов.
    items: [{word, translate}]. Слова без ответа остаются level IS NULL (добьются позже)."""
    if not items:
        return 0
    lines = []
    for it in items:
        tr = it.get("translate", {}) or {}
        hint = ", ".join((tr.get("ru") or tr.get("en") or [])[:3])
        lines.append(f"- {it['word']}" + (f" ({hint})" if hint else ""))
    user = "Классифицируй слова:\n" + "\n".join(lines)
    try:
        data = await ask_json(_CLASSIFY_SYS, user, CLASSIFY_SCHEMA, model, api_key)
    except Exception as e:
        errors.report(e, "classify_batch")
        return 0
    await incr_text_usage(model or LLM_MODEL, api_key)
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    done = 0
    for r in results:
        if not isinstance(r, dict) or not r.get("word"):
            continue
        pid = await get_pool_id(r["word"])
        if not pid:
            continue
        level = r.get("level") if r.get("level") in CEFR_LEVELS else None
        topics = [t for t in (r.get("topics") or []) if t in TOPIC_KEYS]
        if level or topics:
            await set_pool_meta(pid, level=level, topics=topics)
            done += 1
    return done


DESCRIBE_BATCH = int(os.getenv("DESCRIBE_BATCH", "10"))
DESCRIBE_CHECK_SEC = int(os.getenv("DESCRIBE_CHECK_SEC", "10"))  # как часто проверять очередь описаний
_DESCRIBE_SYS = (
    "Ты — преподаватель норвежского. Для каждого норвежского слова дай краткое "
    "(1-2 предложения) понятное описание-толкование на каждом языке: ru, ukr, en, pl, lt. "
    "Верни results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)


async def describe_batch(words, model=None, api_key=None):
    """Пакетные описания (≤DESCRIBE_BATCH слов за один LLM-вызов). Кешируется в БД."""
    if not words:
        return 0
    user = "Опиши слова:\n" + "\n".join(f"- {w}" for w in words)
    try:
        data = await ask_json(_DESCRIBE_SYS, user, DESCRIBE_BATCH_SCHEMA, model, api_key)
    except Exception as e:
        errors.report(e, "describe_batch")
        return 0
    await incr_text_usage(model or LLM_MODEL, api_key)
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    done = 0
    for r in results:
        if not isinstance(r, dict) or not r.get("word"):
            continue
        pid = await get_pool_id(r["word"])
        if not pid:
            continue
        desc = {k: (r.get(k) or "") for k in ("ru", "ukr", "en", "pl", "lt")}
        await set_pool_description(pid, desc)
        done += 1
    return done


async def describe_all_task():
    """Добить описания у всех слов без description — пачками, пока есть слова и бюджет моделей."""
    total = 0
    while True:
        batch = await pool_missing_description(DESCRIBE_BATCH)
        if not batch:
            logger.info(f"describe_all: готово, всего {total}")
            break
        key, model = await _pick_text()
        if not model:
            logger.info(f"describe_all: бюджет исчерпан, всего {total}")
            break
        done = await describe_batch(batch, model, key)
        total += done
        logger.info(f"describe_all: +{done} (всего {total}) via {model}")
        if done == 0:
            break
        await asyncio.sleep(5)  # ≤12 пачек/мин — держимся под лимитом 15 RPM


async def generate_now(n=10):
    """Ручная генерация n новых слов (для Telegram-команды). Возвращает (добавлено, ошибка|None)."""
    key, model = await _pick_text()
    if not model:
        return 0, "бюджет ключей/моделей на сегодня исчерпан"
    letter = random.choices(AUTOFILL_LETTERS, weights=AUTOFILL_LETTER_W, k=1)[0]
    avoid = await get_pool_letter(letter, AUTOFILL_AVOID_SAMPLE)
    nonce = random.randint(1000, 99999)
    prompt = (f"Дай {n} распространённых норвежских слов начинающихся на букву «{letter}». "
              f"Только НОВЫЕ и РАЗНЫЕ. НЕ предлагай уже известные: {', '.join(avoid)}. (вариант {nonce})")
    new_words = []
    try:
        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA, model, key)
        await incr_text_usage(model, key)
        items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for it in items:
            if isinstance(it, dict) and not it.get("error") and it.get("word"):
                existed = await get_pool_id(it["word"])
                pid = await get_or_create_pool(it["word"], normalize_word_item(it))
                if pid:
                    await apply_item_meta(pid, it)
                if not existed:
                    new_words.append(it["word"])
    except Exception as e:
        return 0, errors.report(e, "generate_now").summary
    for nw in new_words:
        await complete_word(nw)  # озвучка сейчас; эмбеддинг добьёт reembed_loop
    return len(new_words), None


async def translate_all_task(max_batches=200):
    """Добить недостающие переводы у слов пула — пачками, пока есть очередь и бюджет.
    Возвращает число дополненных слов."""
    total = 0
    for _ in range(max_batches):
        pend = await translate_pending(TRANSLATE_BATCH)
        if not pend:
            break
        need = []
        for pid, nw, data in pend:
            tr = (data.get("translate") or {})
            missing = [l for l in TRANSLATE_LANGS if not tr.get(l)]
            if not missing:
                await mark_translate_done(pid)
            else:
                need.append((pid, nw, data, missing))
        if not need:
            continue
        key, model = await _pick_text()
        if not model:
            break
        res = await translate_batch([nw for _, nw, _, _ in need], model, key)
        if not res:
            break
        for pid, nw, data, missing in need:
            r = res.get(normalize_word(nw))
            if r:
                tr = dict(data.get("translate") or {})
                for l in missing:
                    vals = [v for v in (r.get(l) or []) if v]
                    if vals:
                        tr[l] = vals
                if tr != (data.get("translate") or {}):
                    await update_pool_translate(pid, tr)
                    total += 1
            await mark_translate_done(pid)
        await asyncio.sleep(2)
    return total


async def describe_loop():
    """Очередь описаний: раз в DESCRIBE_CHECK_SEC проверяем слова без description
    и генерируем одну пачку до DESCRIBE_BATCH (10) слов. Размер пачки = min(кол-во, 10):
    1 слово без описания → 1, 20 → 10 (остаток добьётся на следующих тиках).
    Нагрузка лёгкая (≤6 запросов/мин), чтобы новые слова быстро получали описание.
    Когда очередь пуста — только дешёвый запрос к БД, без обращения к LLM."""
    await asyncio.sleep(15)
    while True:
        if _bg_blocked():
            await asyncio.sleep(20); continue
        try:
            if LLM_API_KEY:
                batch = await pool_missing_description(DESCRIBE_BATCH)
                if batch:
                    key, model = await _pick_text()
                    if model:
                        done = await describe_batch(batch, model, key)
                        logger.info(f"describe_loop: +{done}/{len(batch)} via {model}")
        except Exception as e:
            errors.report(e, "describe_loop")
        await asyncio.sleep(DESCRIBE_CHECK_SEC)


# --- Догенерация недостающих переводов (на все 5 языков) ---
TRANSLATE_LANGS = ["ru", "ukr", "en", "pl", "lt"]
TRANSLATE_BATCH = int(os.getenv("TRANSLATE_BATCH", "10"))
TRANSLATE_CHECK_SEC = int(os.getenv("TRANSLATE_CHECK_SEC", "15"))
_TRANSLATE_SYS = (
    "Ты — переводчик с норвежского (bokmål). Для каждого норвежского слова дай перевод "
    "на 5 языков: ru, ukr, en, pl, lt — по 1-3 варианта (массив строк), без пояснений. "
    "Верни results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)


async def translate_batch(words, model=None, api_key=None):
    """Пакетный перевод списка норвежских слов на 5 языков. Возвращает {norm_word: result}."""
    if not words:
        return {}
    user = "Переведи норвежские слова:\n" + "\n".join(f"- {w}" for w in words)
    try:
        data = await ask_json(_TRANSLATE_SYS, user, TRANSLATE_BATCH_SCHEMA, model, api_key)
    except Exception as e:
        errors.report(e, "translate_batch")
        return {}
    await incr_text_usage(model or LLM_MODEL, api_key)
    out = {}
    for r in ((data or {}).get("results", []) if isinstance(data, dict) else []):
        if isinstance(r, dict) and r.get("word"):
            out[normalize_word(r["word"])] = r
    return out


async def translate_loop():
    """Раз в TRANSLATE_CHECK_SEC берём пачку слов без отметки translate_done.
    Полные (есть все 5 языков) сразу помечаем; неполным догенерируем недостающие
    языки одним LLM-вызовом и обновляем data.translate."""
    await asyncio.sleep(25)
    while True:
        if _bg_blocked():
            await asyncio.sleep(20); continue
        try:
            pend = await translate_pending(TRANSLATE_BATCH)
            if not pend:
                await asyncio.sleep(120)  # очередь пуста — ждём дольше
                continue
            need = []  # (id, norwegian, data, missing_langs)
            for pid, nw, data in pend:
                tr = (data.get("translate") or {})
                missing = [l for l in TRANSLATE_LANGS if not tr.get(l)]
                if not missing:
                    await mark_translate_done(pid)
                else:
                    need.append((pid, nw, data, missing))
            if not need:
                await asyncio.sleep(0.5)  # быстро добиваем «полные»
                continue

            key, model = await _pick_text()
            if not model:
                logger.info("translate_loop: бюджет моделей/ключей исчерпан")
                await asyncio.sleep(300)
                continue
            res = await translate_batch([nw for _, nw, _, _ in need], model, key)
            if not res:
                await asyncio.sleep(TRANSLATE_CHECK_SEC)  # провал запроса — повторим позже
                continue
            done = 0
            for pid, nw, data, missing in need:
                r = res.get(normalize_word(nw))
                if r:
                    tr = dict(data.get("translate") or {})
                    for l in missing:
                        vals = [v for v in (r.get(l) or []) if v]
                        if vals:
                            tr[l] = vals
                    if tr != (data.get("translate") or {}):
                        await update_pool_translate(pid, tr)
                        done += 1
                await mark_translate_done(pid)  # попытка сделана — не зацикливаемся
            logger.info(f"translate_loop: дополнено {done}/{len(need)} via {model}")
        except Exception as e:
            errors.report(e, "translate_loop")
        await asyncio.sleep(TRANSLATE_CHECK_SEC)


# --- Пере-эмбеддинг пула по смыслу (слово+переводы) батчами, с обновлением vec-индекса ---
EMB_SEM_BATCH = min(int(os.getenv("EMB_SEM_BATCH", "100")), 100)  # до 100 текстов за запрос (лимит API)
EMB_SEM_CHECK_SEC = int(os.getenv("EMB_SEM_CHECK_SEC", "2"))


async def reembed_all_task(model="gemini-embedding-2", batch=90, interval=20, report=None, status=None):
    """Добить эмбеддинги у всех слов БЕЗ эмбеддинга моделью `model`, батчами (1 запрос =
    batch слов). Ключи перебираются ПО КРУГУ — один запрос раз в `interval` сек, так каждый
    ключ задействуется поочерёдно (под свой лимит ~100 эмб/мин). Эмбеддит по смыслу
    (слово + переводы). НЕ обнуляет существующие — резюмируемо (полный сброс — /wipeembed).
    report(text) — новый отчёт на каждый батч; status(text) — обновляемая «живая» строка
    (обратный отсчёт до следующего запроса, раз в секунду). На время глушим фон."""
    async def say(t):
        logger.info(t)
        if report:
            try:
                await report(t)
            except Exception:
                pass

    async def countdown(seconds, next_key, remaining):
        """Тикаем раз в секунду в Telegram (редактируя одну строку) до следующего запроса."""
        if not status or seconds <= 0:
            await asyncio.sleep(max(0, seconds))
            return
        for s in range(int(seconds), 0, -1):
            try:
                await status(f"⏳ Следующий запрос (k{next_key}) через {s}с · осталось {remaining}")
            except Exception:
                pass
            await asyncio.sleep(1)

    # Объединяем пулы ключей (это одни и те же Google-ключи) — перебираем их по кругу.
    keys = list(dict.fromkeys((EMBED_API_KEYS or []) + (LLM_API_KEYS or []))) or [None]
    prev_paused = RUNTIME["paused"]
    RUNTIME["paused"] = True  # на время bulk-операции глушим фон, чтобы не мешал/не жёг квоту
    try:
        total = await count_missing_embedding()
        if not total:
            await say("🔁 Все слова уже с эмбеддингом — нечего добивать.")
            return 0
        await say(f"🔁 Добиваю эмбеддинги: {total} слов без вектора, модель {model}, по {batch} за "
                  f"запрос, ключей {len(keys)} по кругу раз в {interval}с. Фон на паузе.")
        done = bn = last_id = ki = fails = q429 = stall = 0
        while True:
            rows = await missing_embedding_data(batch, after_id=last_id)
            if not rows:
                break
            bn += 1
            key = keys[ki]; used = ki
            ki = (ki + 1) % len(keys)  # round-robin: следующий запрос — следующим ключом
            items = [(pid, semantic_embed_text(data)) for pid, data in rows]
            items = [(pid, t) for pid, t in items if t]  # только со смысловым текстом
            if items:
                try:
                    vecs = await embed_texts([t for _, t in items], model, key, raise_on_error=True)
                except Exception as e:
                    info = errors.classify(e)
                    if info.kind == errors.QUOTA:
                        q429 += 1
                        if q429 >= len(keys):  # прошли все ключи по кругу — все в 429
                            q429 = 0; stall += 1
                            if stall >= 5:
                                await say(f"⛔ Все ключи в 429 слишком долго — прерываю. Сделано {done}/{total}, "
                                          f"осталось {await count_missing_embedding()}. Запусти /reembed2 позже.")
                                break
                            await say(f"  ⏳ Все ключи 429 — пауза 120с (попытка {stall}/5)")
                            await asyncio.sleep(120)
                        else:
                            await say(f"  🔑 429 на k{used} — пробую следующий ключ через {interval}с")
                            await asyncio.sleep(interval)
                        continue  # курсор НЕ двигаем — повторим тот же батч следующим ключом
                    fails += 1
                    if fails >= 5:
                        await say(f"⛔ Прерываю: 5 ошибок подряд ({info.summary}). Сделано {done}/{total}, "
                                  f"осталось {await count_missing_embedding()}. /reembed2 позже.")
                        break
                    await say(f"  ⚠️ батч {bn} (k{used}): {info.kind} ({fails}/5) — повтор через {interval}с")
                    await asyncio.sleep(interval)
                    continue
                if not (vecs and len(vecs) == len(items)):
                    fails += 1
                    if fails >= 5:
                        await say(f"⛔ Прерываю: 5 пустых ответов подряд. Сделано {done}/{total}. /reembed2 позже.")
                        break
                    await asyncio.sleep(interval)
                    continue
                for (pid, _), vec in zip(items, vecs):
                    await set_pool_embedding(pid, encode_emb(vec))
                    await mark_sem_embed(pid)
                done += len(items)
            fails = q429 = stall = 0  # успешный батч — сбрасываем счётчики
            last_id = rows[-1][0]  # курсор вперёд — гарантирует завершение
            remaining = await count_missing_embedding()
            if items:  # был реальный запрос — отчитываемся и выдерживаем интервал с отсчётом
                await say(f"  ✅ батч {bn} (k{used}): +{len(items)} · осталось {remaining}")
                if remaining > 0:
                    await countdown(interval, ki, remaining)  # ki уже указывает на следующий ключ
    finally:
        RUNTIME["paused"] = prev_paused
    if status:
        try:
            await status(f"🏁 Пере-эмбеддинг завершён: {done}/{total}.")
        except Exception:
            pass
    await say(f"🏁 Готово: пере-эмбеддено {done}/{total} на {model}. Фон возобновлён.")
    return done


async def reembed_loop():
    """Пересчитываем эмбеддинги пула по смысловому тексту (слово + все переводы) —
    БАТЧАМИ до 100 за один запрос (1 запрос = 1 единица квоты, лимит 1000/день по
    запросам). set_pool_embedding обновляет и vec-индекс. Флаг word_pool.emb_sem."""
    await asyncio.sleep(20)
    while True:
        if _bg_blocked():
            await asyncio.sleep(20); continue
        try:
            batch = await sem_embed_pending(EMB_SEM_BATCH)
            if not batch:
                await asyncio.sleep(300)  # всё пересчитано — ждём дольше
                continue
            key, model = await _pick_emb()
            if not model:
                await asyncio.sleep(900)
                continue
            # слова без текста (нет переводов) — просто помечаем
            items = []
            for pid, data in batch:
                text = semantic_embed_text(data)
                if text:
                    items.append((pid, text))
                else:
                    await mark_sem_embed(pid)
            if not items:
                continue
            vecs = await embed_texts([t for _, t in items], model, key)
            if vecs and len(vecs) == len(items):
                for (pid, _), vec in zip(items, vecs):
                    await set_pool_embedding(pid, encode_emb(vec))  # обновляет и vec-индекс
                    await mark_sem_embed(pid)
                logger.info(f"reembed batch: +{len(items)} via {model}")
                await asyncio.sleep(EMB_SEM_CHECK_SEC)
            else:
                await asyncio.sleep(60)  # 429/ошибка батча — переждём (RPM/квота), повторим
        except Exception as e:
            errors.report(e, "reembed_loop")
            await asyncio.sleep(30)


async def autofill_loop():
    await asyncio.sleep(15)
    i = 0
    while True:
        if _bg_blocked(check_autofill=True):
            await asyncio.sleep(20); continue
        night = _is_night()
        interval = AUTOFILL_NIGHT_INTERVAL_SEC if night else AUTOFILL_INTERVAL_SEC
        try:
            # фон работает только в простое — после N секунд без активности юзеров
            idle = seconds_idle()
            if idle < AUTOFILL_IDLE_SEC:
                await asyncio.sleep(min(AUTOFILL_IDLE_SEC - idle + 1, 60))
                continue
            if LLM_API_KEY:
                # Ротация ключей × моделей по суточным лимитам (RPD — на ключ для каждой
                # модели). Озвучка (edge) бесплатна — добиваем всегда; текст/эмбеддинги —
                # пока есть бюджет хоть у одной пары (ключ × модель).
                text_key, text_model = await _pick_text()
                emb_key, emb_model = await _pick_emb()
                # пока идёт пере-эмбеддинг — эмбеддингом занимается ТОЛЬКО батч-цикл reembed_loop
                # (autofill не долбит по одному, чтобы не жечь дневной лимит запросов и RPM)
                emb_miss = (await pool_missing_embedding(1)) if (emb_model and not await sem_embed_pending(1)) else []
                tts_miss = await pool_missing_tts(1)
                unclassified = await pool_missing_meta(CLASSIFY_BATCH) if (text_model and not emb_miss and not tts_miss) else None
                if not text_model and not emb_model and not tts_miss:
                    # квота всех моделей на сегодня исчерпана, доделывать нечего — ждём подольше
                    await asyncio.sleep(600)
                    continue
                if emb_miss or tts_miss:
                    w = (emb_miss or tts_miss)[0]
                    await complete_word(w, emb_model=emb_model, emb_key=emb_key)  # эмбеддинг (если есть бюджет) + озвучка
                    logger.info(f"autofill: completed '{w}'")
                elif unclassified:
                    done = await classify_batch(unclassified, model=text_model, api_key=text_key)  # пачка: уровень + темы
                    logger.info(f"autofill: classified {done}/{len(unclassified)} via {text_model}")
                elif text_model and not await sem_embed_pending(1):
                    # Пока не пересчитаны ВСЕ эмбеддинги по смыслу — новые слова не добавляем
                    # (чтобы пул не рос во время пере-эмбеддинга и статистика была понятной).
                    i += 1
                    # Генерация новых слов — основной приоритет. Описания не догоняем
                    # фоном (генерятся по запросу при открытии), чтобы пул рос быстрее.
                    nonce = random.randint(1000, 99999)
                    # 2 из 3 циклов — по букве (системно вычищаем алфавит), 1 из 3 — по теме.
                    if i % 3 == 0:
                        topic = AUTOFILL_TOPICS[(i // 3) % len(AUTOFILL_TOPICS)]
                        avoid = await get_pool_sample(AUTOFILL_AVOID_SAMPLE)
                        constraint = f"на тему: {topic}"
                        label = f"topic='{topic}'"
                    else:
                        letter = random.choices(AUTOFILL_LETTERS, weights=AUTOFILL_LETTER_W, k=1)[0]
                        avoid = await get_pool_letter(letter, AUTOFILL_AVOID_SAMPLE)
                        constraint = f"начинающихся на букву «{letter}»"
                        label = f"letter='{letter}'"
                    avoid_s = ", ".join(avoid)
                    prompt = (f"Дай {AUTOFILL_BATCH} распространённых норвежских слов {constraint}. "
                              f"Только НОВЫЕ и РАЗНЫЕ. НЕ предлагай уже известные: {avoid_s}. (вариант {nonce})")
                    new_words = []
                    try:
                        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA, text_model, text_key)
                        await incr_text_usage(text_model, text_key)
                        items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                        for it in items:
                            if isinstance(it, dict) and not it.get("error") and it.get("word"):
                                existed = await get_pool_id(it["word"])
                                pid = await get_or_create_pool(it["word"], normalize_word_item(it))
                                if pid:
                                    await apply_item_meta(pid, it)
                                if not existed:
                                    new_words.append(it["word"])
                    except Exception as e:
                        errors.report(e, "autofill generate")
                    for nw in new_words:
                        await complete_word(nw, emb_model=emb_model, emb_key=emb_key)  # эмбеддинг (если бюджет) + озвучка
                    logger.info(f"autofill: {label} new={len(new_words)} via {text_model}")
        except Exception as e:
            errors.report(e, "autofill_loop")
            await asyncio.sleep(120)
            continue
        await asyncio.sleep(interval)
