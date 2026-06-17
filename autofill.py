import os
import random
import asyncio
from datetime import datetime
from config import logger
import errors
import notify
import runtime  # рантайм-флаги паузы фоновых задач (с админ-страницы)
from activity import seconds_idle
from db import (
    get_pool_id, get_pool_by_id, set_pool_embedding, get_pool_tts, set_pool_tts,
    get_or_create_pool, set_pool_meta, set_pool_description,
    pool_missing_embedding, pool_missing_tts, pool_missing_meta, pool_missing_description,
    translate_pending, mark_translate_done, update_pool_translate, normalize_word,
    sem_embed_pending, mark_sem_embed,
    get_pool_sample, get_pool_letter, pos_missing_forms, set_pool_forms,
)
import llm  # текст/эмбеддинги через key-free API: ask_json/embed_texts/text_budget_left/...
from llm import (
    embed_texts, encode_emb, ask_json, semantic_embed_text,
    WORDS_SCHEMA, CLASSIFY_SCHEMA, DESCRIBE_BATCH_SCHEMA, TRANSLATE_BATCH_SCHEMA,
    NOUN_FORMS_SCHEMA, VERB_FORMS_SCHEMA, ADJ_FORMS_SCHEMA,
    TOPIC_TAGS, TOPIC_KEYS, CEFR_LEVELS, normalize_word_item, apply_item_meta,
)
from tts import synth_tts, _tts_lock
from task import task

# --- Авто-заполнение общего пула в рамках суточного бюджета ("свободная" квота) ---
AUTOFILL_ENABLED = os.getenv("AUTOFILL_ENABLED", "false").lower() == "true"
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
# Профили моделей и ротация ключей/квоты — целиком внутри llm.py (key-free API):
# llm.ask_json(purpose="autofill"), llm.embed_texts(...), llm.text_available("autofill"),
# llm.embed_available(), llm.text_enabled().


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



async def complete_batch(words):
    """Доделать ПАЧКУ слов: эмбеддинги одним запросом (для тех, у кого вектора нет) +
    озвучка каждого (edge, бесплатно). Ключ/модель эмбеддинга — внутри llm.embed_texts.
    Возвращает число посчитанных эмбеддингов."""
    pending = []  # (pid, текст-для-эмбеддинга) — только у кого вектора ещё нет
    if llm.embed_enabled() and not runtime.PAUSED["embed"]:
        for w in words:
            pid = await get_pool_id(w)
            if not pid:
                continue
            p = await get_pool_by_id(pid)
            if p and not p.get("embedding"):
                pending.append((pid, semantic_embed_text(p["data"]) or w))
    n = 0
    if pending:
        vecs = await embed_texts([t for _, t in pending])
        if vecs and len(vecs) == len(pending):
            for (pid, _), vec in zip(pending, vecs):
                await set_pool_embedding(pid, encode_emb(vec))
                await mark_sem_embed(pid)
            n = len(pending)
    for w in words:  # озвучка — по одному (edge бесплатный, без квоты)
        if not await get_pool_tts(w):
            async with _tts_lock:
                try:
                    mp3 = await synth_tts(w)
                except Exception as e:
                    mp3 = None
                    logger.warning(f"complete_batch tts '{w}': {e}")
                if mp3:
                    await set_pool_tts(w, mp3)
    return n


_CLASSIFY_SYS = (
    "Ты — лингвист-методист по норвежскому (bokmål). Для каждого слова определи: "
    "уровень CEFR (A1, A2, B1, B2, C1 или C2 — по частотности и сложности) и 1-3 темы "
    "СТРОГО из этого списка ключей:\n"
    + "; ".join(f"{k} ({v})" for k, v in TOPIC_TAGS.items())
    + ".\nВерни массив results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)


async def classify_batch(items):
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
        data = await ask_json(_CLASSIFY_SYS, user, CLASSIFY_SCHEMA, purpose="autofill",
                              label=f"классификация слов ({len(items)})")
    except Exception as e:
        errors.report(e, "classify_batch")
        return 0
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    # как и в describe — сопоставляем ответ ВХОДНОМУ слову (нормализованно/позиционно),
    # чтобы «исправленное» моделью написание не оставляло слово вечно неклассифицированным.
    words = [it["word"] for it in items]
    norm_to_word = {normalize_word(w): w for w in words}
    positional = len(results) == len(words)
    done, seen = 0, set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        src = norm_to_word.get(normalize_word(r.get("word") or ""))
        if src is None and positional:
            src = words[i]
        if not src or src in seen:
            continue
        seen.add(src)
        pid = await get_pool_id(src)
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


async def describe_batch(words):
    """Пакетные описания (≤DESCRIBE_BATCH слов за один LLM-вызов). Кешируется в БД."""
    if not words:
        return 0
    user = "Опиши слова:\n" + "\n".join(f"- {w}" for w in words)
    try:
        data = await ask_json(_DESCRIBE_SYS, user, DESCRIBE_BATCH_SCHEMA, purpose="autofill",
                              label=f"описания слов ({len(words)})")
    except Exception as e:
        errors.report(e, "describe_batch")
        return 0
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    # Сохраняем описание по ВХОДНОМУ слову (что спрашивали), а не по тому, как модель его
    # переписала: по нормализованному имени, иначе позиционно. Иначе слова с «кривым»
    # написанием (модель его «исправляет») никогда не сохраняются и крутят очередь вечно.
    norm_to_word = {normalize_word(w): w for w in words}
    positional = len(results) == len(words)
    done, seen = 0, set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        src = norm_to_word.get(normalize_word(r.get("word") or ""))
        if src is None and positional:
            src = words[i]
        if not src or src in seen:
            continue
        seen.add(src)
        pid = await get_pool_id(src)
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
        if not llm.text_available("autofill"):
            logger.info(f"describe_all: бюджет исчерпан, всего {total}")
            break
        done = await describe_batch(batch)
        total += done
        logger.info(f"describe_all: +{done} (всего {total})")
        if done == 0:
            break
        await asyncio.sleep(5)  # ≤12 пачек/мин — держимся под лимитом 15 RPM


async def describe_loop():
    """Очередь описаний: раз в DESCRIBE_CHECK_SEC проверяем слова без description
    и генерируем одну пачку до DESCRIBE_BATCH (10) слов. Размер пачки = min(кол-во, 10):
    1 слово без описания → 1, 20 → 10 (остаток добьётся на следующих тиках).
    Нагрузка лёгкая (≤6 запросов/мин), чтобы новые слова быстро получали описание.
    Когда очередь пуста — только дешёвый запрос к БД, без обращения к LLM."""
    await asyncio.sleep(15)
    while True:
        if runtime.PAUSED["describe"]:
            await asyncio.sleep(20); continue
        try:
            if llm.text_enabled():
                batch = await pool_missing_description(DESCRIBE_BATCH)
                if batch and llm.text_available("autofill"):
                    done = await describe_batch(batch)
                    logger.info(f"describe_loop: +{done}/{len(batch)}")
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


async def translate_batch(words):
    """Пакетный перевод списка норвежских слов на 5 языков. Возвращает {norm_word: result}."""
    if not words:
        return {}
    user = "Переведи норвежские слова:\n" + "\n".join(f"- {w}" for w in words)
    try:
        data = await ask_json(_TRANSLATE_SYS, user, TRANSLATE_BATCH_SCHEMA, purpose="autofill",
                              label=f"дополнение переводов ({len(words)})")
    except Exception as e:
        errors.report(e, "translate_batch")
        return {}
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

            if not llm.text_available("autofill"):
                logger.info("translate_loop: бюджет моделей/ключей исчерпан")
                await asyncio.sleep(300)
                continue
            res = await translate_batch([nw for _, nw, _, _ in need])
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
            logger.info(f"translate_loop: дополнено {done}/{len(need)}")
        except Exception as e:
            errors.report(e, "translate_loop")
        await asyncio.sleep(TRANSLATE_CHECK_SEC)


# --- Пере-эмбеддинг пула по смыслу (слово+переводы) батчами, с обновлением vec-индекса ---
EMB_SEM_BATCH = min(int(os.getenv("EMB_SEM_BATCH", "100")), 100)  # до 100 текстов за запрос (лимит API)
EMB_SEM_CHECK_SEC = int(os.getenv("EMB_SEM_CHECK_SEC", "2"))


async def reembed_loop():
    """Пересчитываем эмбеддинги пула по смысловому тексту (слово + все переводы) —
    БАТЧАМИ до 100 за один запрос (1 запрос = 1 единица квоты, лимит 1000/день по
    запросам). set_pool_embedding обновляет и vec-индекс. Флаг word_pool.emb_sem."""
    await asyncio.sleep(20)
    while True:
        if runtime.PAUSED["embed"]:
            await asyncio.sleep(20); continue
        try:
            batch = await sem_embed_pending(EMB_SEM_BATCH)
            if not batch:
                await asyncio.sleep(300)  # всё пересчитано — ждём дольше
                continue
            if not llm.embed_available():
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
            vecs = await embed_texts([t for _, t in items])
            if vecs and len(vecs) == len(items):
                for (pid, _), vec in zip(items, vecs):
                    await set_pool_embedding(pid, encode_emb(vec))  # обновляет и vec-индекс
                    await mark_sem_embed(pid)
                logger.info(f"reembed batch: +{len(items)}")
                await asyncio.sleep(EMB_SEM_CHECK_SEC)
            else:
                await asyncio.sleep(60)  # 429/ошибка батча — переждём (RPM/квота), повторим
        except Exception as e:
            errors.report(e, "reembed_loop")
            await asyncio.sleep(30)


# --- Грамматические формы по части речи (фоновая очередь, отдельный промпт на POS) ---
FORMS_BATCH = int(os.getenv("FORMS_BATCH", "20"))
FORMS_CHECK_SEC = int(os.getenv("FORMS_CHECK_SEC", "12"))
_FORMS = {
    "noun": dict(
        schema=NOUN_FORMS_SCHEMA, fields=["gender", "def_sg", "indef_pl", "def_pl"],
        sys=("Ты — эксперт по существительным норвежского (bokmål). Для каждого слова дай ТОЧНЫЕ "
             "формы (включая нерегулярные): gender (en/ei/et; женский род давай 'ei'), def_sg "
             "(опр. ед.), indef_pl (неопр. мн.), def_pl (опр. мн.). Поле word — ровно как на входе.")),
    "verb": dict(
        schema=VERB_FORMS_SCHEMA, fields=["present", "past", "perfect"],
        sys=("Ты — эксперт по глаголам норвежского (bokmål). Для каждого инфинитива дай ТОЧНЫЕ "
             "формы (включая сильные/нерегулярные): present (презенс), past (претеритум), perfect "
             "(перфект с 'har'). Поле word — ровно как на входе.")),
    "adjective": dict(
        schema=ADJ_FORMS_SCHEMA, fields=["comparative", "superlative", "neuter", "plural"],
        sys=("Ты — эксперт по прилагательным норвежского (bokmål). Для каждого дай ТОЧНЫЕ формы "
             "(включая нерегулярные степени): comparative, superlative, neuter (средний род -t), "
             "plural (мн./опр. -e). Поле word — ровно как на входе.")),
}


async def forms_batch(category, rows):
    """rows: [(pid, norwegian, data)]. Генерит грамм. формы пачкой (temperature=0) и сохраняет.
    Сопоставление по ВХОДНОМУ слову (как в describe) — модель могла переписать. Возвращает кол-во."""
    cfg = _FORMS[category]
    user = "Слова:\n" + "\n".join(f"- {nw}" for _, nw, _ in rows)
    try:
        data = await ask_json(cfg["sys"], user, cfg["schema"], purpose="autofill",
                              label=f"формы {category} ({len(rows)})", temperature=0)
    except Exception as e:
        errors.report(e, f"forms_batch({category})")
        return 0
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    norm_to_pid = {normalize_word(nw): pid for pid, nw, _ in rows}
    positional = len(results) == len(rows)
    done, seen = 0, set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        pid = norm_to_pid.get(normalize_word(r.get("word") or ""))
        if pid is None and positional:
            pid = rows[i][0]
        if not pid or pid in seen:
            continue
        seen.add(pid)
        forms = {f: r[f].strip() for f in cfg["fields"]
                 if isinstance(r.get(f), str) and r[f].strip() and r[f].strip().lower() not in ("-", "null")}
        # сохраняем даже пустые формы (только pos) — чтобы слово считалось «обработанным» и не зациклилось
        await set_pool_forms(pid, {"pos": category, **forms})
        done += 1
    return done


async def forms_loop():
    """Фоновая догенерация грамматических форм. Чередуем части речи (свой промпт на каждую),
    по FORMS_BATCH слов за запрос. Слова без подходящей POS форм не получают (наречия/фразы)."""
    await asyncio.sleep(30)
    cats = list(_FORMS.keys())
    i = 0
    while True:
        if runtime.PAUSED["forms"]:
            await asyncio.sleep(20); continue
        try:
            if llm.text_enabled() and llm.text_available("autofill"):
                cat = cats[i % len(cats)]; i += 1
                rows = await pos_missing_forms(cat, FORMS_BATCH)
                if rows:
                    done = await forms_batch(cat, rows)
                    logger.info(f"forms_loop[{cat}]: +{done}/{len(rows)}")
        except Exception as e:
            errors.report(e, "forms_loop")
        await asyncio.sleep(FORMS_CHECK_SEC)


async def autofill_loop():
    await asyncio.sleep(15)
    i = 0
    while True:
        if runtime.PAUSED["autofill"]:
            await asyncio.sleep(20); continue
        night = _is_night()
        interval = AUTOFILL_NIGHT_INTERVAL_SEC if night else AUTOFILL_INTERVAL_SEC
        try:
            # фон работает только в простое — после N секунд без активности юзеров
            idle = seconds_idle()
            if idle < AUTOFILL_IDLE_SEC:
                await asyncio.sleep(min(AUTOFILL_IDLE_SEC - idle + 1, 60))
                continue
            if llm.text_enabled():
                # Есть ли ещё суточный бюджет (квота/ключи) на текст и эмбеддинги. Само
                # распределение по ключам×моделям — внутри llm. Озвучка (edge) бесплатна.
                text_ok = llm.text_available("autofill")
                emb_ok = llm.embed_available()
                # пока идёт пере-эмбеддинг — эмбеддингом занимается ТОЛЬКО батч-цикл reembed_loop
                # (autofill не долбит по одному, чтобы не жечь дневной лимит запросов и RPM)
                emb_miss = (await pool_missing_embedding(10)) if (emb_ok and not await sem_embed_pending(1)) else []
                tts_miss = await pool_missing_tts(10)
                unclassified = await pool_missing_meta(CLASSIFY_BATCH) if (text_ok and not emb_miss and not tts_miss) else None
                if not text_ok and not emb_ok and not tts_miss:
                    # квота на сегодня исчерпана, доделывать нечего — ждём подольше
                    await asyncio.sleep(600)
                    continue
                if emb_miss or tts_miss:
                    words = emb_miss or tts_miss  # пачка до 10: эмбеддинг одним запросом + озвучка
                    n = await complete_batch(words)
                    logger.info(f"autofill: completed batch {len(words)} (emb +{n})")
                elif unclassified:
                    done = await classify_batch(unclassified)  # пачка: уровень + темы
                    logger.info(f"autofill: classified {done}/{len(unclassified)}")
                elif text_ok and not await sem_embed_pending(1):
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
                    new_words, dup_words = [], []
                    ok = True
                    try:
                        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA,
                                              purpose="autofill", label=f"автозаполнение ({label})")
                        items = data.get("words", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                        for it in items:
                            if isinstance(it, dict) and not it.get("error") and it.get("word"):
                                existed = await get_pool_id(it["word"])
                                pid = await get_or_create_pool(it["word"], normalize_word_item(it))
                                if pid:
                                    await apply_item_meta(pid, it)
                                (dup_words if existed else new_words).append(it["word"])
                    except Exception as e:
                        ok = False
                        errors.report(e, "autofill generate")  # ошибку уже отчитали — «пусто» не шлём
                    if new_words:
                        await complete_batch(new_words)  # пачкой
                    logger.info(f"autofill: {label} new={len(new_words)} dup={len(dup_words)}")
                    # Разбивка в ленту: LLM иногда повторяет известные слова (несмотря на стоп-лист),
                    # а иногда не находит новых вовсе — тогда честно пишем «пусто».
                    if ok and not new_words and not dup_words:
                        notify.feed(f"🆕 автозаполнение ({label}): пусто — модель не нашла новых слов "
                                    f"(просили {AUTOFILL_BATCH}, все распространённые уже в пуле)")
                    elif ok:
                        msg = (f"🆕 автозаполнение ({label}): просили {AUTOFILL_BATCH} · "
                               f"новых {len(new_words)} · повтор {len(dup_words)}")
                        if new_words:
                            msg += f"\n  ➕ {', '.join(new_words)}"
                        if dup_words:
                            msg += f"\n  ↩️ уже были: {', '.join(dup_words)}"
                        notify.feed(msg)
        except Exception as e:
            errors.report(e, "autofill_loop")
            await asyncio.sleep(120)
            continue
        await asyncio.sleep(interval)
