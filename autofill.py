import os
import random
import asyncio
from datetime import datetime
from config import logger
from activity import seconds_idle
from db import (
    get_pool_id, get_pool_by_id, set_pool_embedding, get_pool_tts, set_pool_tts,
    get_or_create_pool, set_pool_meta, pool_missing_embedding, pool_missing_tts,
    pool_missing_meta, get_pool_sample, get_pool_letter, incr_usage, get_usage,
)
from llm import (
    LLM_API_KEY, LLM_MODEL, EMBED_MODEL, embed_text, encode_emb, ask_json, WORDS_SCHEMA,
    CLASSIFY_SCHEMA, TOPIC_TAGS, TOPIC_KEYS, CEFR_LEVELS,
    normalize_word_item, apply_item_meta,
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


def _today():
    return datetime.utcnow().strftime("%Y-%m-%d")


async def _pick_model(models, kind):
    """Первая модель из списка, у которой ещё есть суточный бюджет. None — все исчерпаны."""
    today = _today()
    for m, budget in models:
        if (await get_usage(f"{today}:{kind}:{m}")) < budget:
            return m
    return None


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


async def complete_word(w, emb_model=None):
    """Доделать слово: эмбеддинг (если задана модель emb_model с бюджетом) + озвучка (всегда, edge бесплатный)."""
    pid = await get_pool_id(w)
    if emb_model and pid:
        p = await get_pool_by_id(pid)
        if p and not p.get("embedding"):
            vec = await embed_text(w, model=emb_model)
            if vec:
                await set_pool_embedding(pid, encode_emb(vec))
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


async def classify_batch(items, model=None):
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
        data = await ask_json(_CLASSIFY_SYS, user, CLASSIFY_SCHEMA, model)
    except Exception as e:
        logger.warning(f"classify_batch: {e}")
        return 0
    await incr_usage(_today() + ":text:" + (model or LLM_MODEL))
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


async def autofill_loop():
    await asyncio.sleep(15)
    i = 0
    while True:
        night = _is_night()
        interval = AUTOFILL_NIGHT_INTERVAL_SEC if night else AUTOFILL_INTERVAL_SEC
        try:
            # фон работает только в простое — после N секунд без активности юзеров
            idle = seconds_idle()
            if idle < AUTOFILL_IDLE_SEC:
                await asyncio.sleep(min(AUTOFILL_IDLE_SEC - idle + 1, 60))
                continue
            if LLM_API_KEY:
                # Ротация моделей по суточным лимитам (у каждой свой RPD). Озвучка (edge)
                # бесплатна — её добиваем всегда; текст/эмбеддинги — пока есть бюджет у моделей.
                text_model = await _pick_model(TEXT_MODELS, "text")
                emb_model = await _pick_model(EMBED_MODELS, "emb")
                emb_miss = (await pool_missing_embedding(1)) if emb_model else []
                tts_miss = await pool_missing_tts(1)
                unclassified = await pool_missing_meta(CLASSIFY_BATCH) if (text_model and not emb_miss and not tts_miss) else None
                if not text_model and not emb_model and not tts_miss:
                    # квота всех моделей на сегодня исчерпана, доделывать нечего — ждём подольше
                    await asyncio.sleep(600)
                    continue
                if emb_miss or tts_miss:
                    w = (emb_miss or tts_miss)[0]
                    await complete_word(w, emb_model=emb_model)  # эмбеддинг (если есть бюджет) + озвучка
                    logger.info(f"autofill: completed '{w}'")
                elif unclassified:
                    done = await classify_batch(unclassified, model=text_model)  # пачка: уровень + темы
                    logger.info(f"autofill: classified {done}/{len(unclassified)} via {text_model}")
                elif text_model:
                    i += 1
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
                        data = await ask_json(task, f"Текст запроса от пользователя: >>{prompt}<<", WORDS_SCHEMA, text_model)
                        await incr_usage(_today() + ":text:" + text_model)
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
                        logger.warning(f"autofill generate: {e}")
                    for nw in new_words:
                        await complete_word(nw, emb_model=emb_model)  # эмбеддинг (если бюджет) + озвучка
                    logger.info(f"autofill: {label} new={len(new_words)} via {text_model}")
        except Exception as e:
            logger.warning(f"autofill error: {e}")
            await asyncio.sleep(120)
            continue
        await asyncio.sleep(interval)
