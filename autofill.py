import os
import json
import random
import asyncio
from datetime import datetime
from config import logger
import errors
import notify
import runtime  # рантайм-флаги паузы фоновых задач (с админ-страницы)
from activity import seconds_idle
from db import (
    get_pool_by_id, set_pool_embedding, get_pool_tts, set_pool_tts,
    get_or_create_pool, set_pool_meta, set_pool_description,
    pool_missing_embedding, pool_missing_tts, pool_missing_meta, pool_missing_description,
    translate_pending, mark_translate_done, update_pool_translate, normalize_word,
    yo_pending, mark_yo_done,
    sem_embed_pending, mark_sem_embed,
    pos_missing_forms, set_pool_forms,
    nouns_missing_countability, merge_pool_forms,
    pos_uncategorized, set_pool_pos, get_pool_words_by_names,
    dedup_pending, mark_dedup, pool_usage_count, nearest_other, merge_pool_words,
    freq_pending, set_pool_freq_bulk,
)
import llm  # текст/эмбеддинги через key-free API: ask_json/embed_texts/text_budget_left/...
from llm import (
    embed_texts, encode_emb, ask_json, semantic_embed_text,
    WORDS_SCHEMA, CLASSIFY_SCHEMA, DESCRIBE_BATCH_SCHEMA, TRANSLATE_BATCH_SCHEMA,
    NOUN_FORMS_SCHEMA, VERB_FORMS_SCHEMA, ADJ_FORMS_SCHEMA, COUNTABLE_SCHEMA, POS_REFINE_SCHEMA, POS_KEYS,
    TOPIC_TAGS, TOPIC_KEYS, CEFR_LEVELS, LANG_NAMES, normalize_word_item, apply_item_meta,
    DEDUP_SCHEMA,
)
from tts import synth_tts, _tts_lock
from langs import LANG_CODES, LANGS_CSV
from task import task
from autofill_enrich import (  # noqa: E402,F401 — реэкспорт для воркеров
    classify_batch, describe_batch, translate_batch, forms_batch, _FORMS,
)
from autofill_wordgen import (  # noqa: E402,F401 — реэкспорт для воркеров/роутеров
    ai_game_words, generate_set_words, words_from_image, words_from_text, words_from_list, restore_yo,
    _persist_word_items, _embed_new,
)

# --- Фоновая доделка недостающего (эмбеддинг/озвучка/классификация).
# Генерация НОВЫХ слов LLM-ом выпилена (3.07.2026): база собирается из
# человеческих источников (Lexin/ordbank), фоновому генератору тут делать нечего.
AUTOFILL_INTERVAL_SEC = int(os.getenv("AUTOFILL_INTERVAL_SEC", "150"))  # одно слово целиком раз в N сек (день)
AUTOFILL_IDLE_SEC = int(os.getenv("AUTOFILL_IDLE_SEC", "300"))  # фон только после N сек простоя
# Ночной режим — агрессивнее (когда никто не пользуется)
AUTOFILL_NIGHT_INTERVAL_SEC = int(os.getenv("AUTOFILL_NIGHT_INTERVAL_SEC", "6"))   # ~10 операций/мин
AUTOFILL_NIGHT_START = int(os.getenv("AUTOFILL_NIGHT_START", "2"))    # локальный час начала ночи
AUTOFILL_NIGHT_END = int(os.getenv("AUTOFILL_NIGHT_END", "7"))        # локальный час конца ночи
AUTOFILL_TZ_OFFSET = int(os.getenv("AUTOFILL_TZ_OFFSET", "2"))        # сдвиг от UTC (Норвегия летом = +2)
CLASSIFY_BATCH = int(os.getenv("CLASSIFY_BATCH", "50"))  # слов на один LLM-вызов классификации
# Профили моделей и ротация ключей/квоты — целиком внутри llm.py (key-free API):
# llm.ask_json(purpose="autofill"), llm.embed_texts(...), llm.text_available("autofill"),
# llm.embed_available(), llm.text_enabled().


def _is_night():
    h = (datetime.utcnow().hour + AUTOFILL_TZ_OFFSET) % 24
    s, e = AUTOFILL_NIGHT_START, AUTOFILL_NIGHT_END
    return (s <= h < e) if s <= e else (h >= s or h < e)


async def tts_backfill(words):
    """Озвучка (edge, бесплатно) — по СЛОВУ: один mp3 на написание, все омонимы делят его
    (set_pool_tts обновляет все записи написания). pos не нужен — звучание от него не зависит.
    words — [norwegian]. Возвращает число сгенерённых озвучек."""
    made, seen = 0, set()
    for w in words:
        key = normalize_word(w)
        if key in seen:
            continue
        seen.add(key)
        if not await get_pool_tts(w):
            async with _tts_lock:
                try:
                    mp3 = await synth_tts(w)
                except Exception as e:
                    mp3 = None
                    logger.warning(f"tts_backfill '{w}': {e}")
                if mp3:
                    await set_pool_tts(w, mp3)
                    made += 1
    return made


async def embed_backfill(items):
    """Эмбеддинги пачкой — строго по ID. Омонимы (один norwegian → несколько записей с разным
    pos) имеют КАЖДЫЙ свой вектор: get_pool_id без pos попал бы в старшую запись, вектор нужной
    остался бы NULL и запись переизбиралась бы вечно (инцидент autofill-homograph-requeue —
    слив квоты). Поэтому items — ТОЛЬКО [(pid, norwegian)]; резолва по слову тут нет вовсе.
    Возвращает число посчитанных векторов."""
    if not (llm.embed_enabled() and not runtime.PAUSED["embed"]):
        return 0
    pending, seen = [], set()   # (pid, текст) — только у кого вектора ещё нет
    for pid, w in items:
        if not pid or pid in seen:
            continue
        seen.add(pid)
        p = await get_pool_by_id(pid)
        if p and not p.get("embedding"):
            pending.append((pid, semantic_embed_text(p["data"]) or w))
    if not pending:
        return 0
    vecs = await embed_texts([t for _, t in pending])
    if vecs and len(vecs) == len(pending):
        for (pid, _), vec in zip(pending, vecs):
            await set_pool_embedding(pid, encode_emb(vec))
            await mark_sem_embed(pid)
        return len(pending)
    return 0






DESCRIBE_BATCH = int(os.getenv("DESCRIBE_BATCH", "10"))
DESCRIBE_CHECK_SEC = int(os.getenv("DESCRIBE_CHECK_SEC", "4"))  # как часто проверять очередь описаний (5 ключей → можно чаще)




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
        await asyncio.sleep(2)  # 5 ключей × 15 RPM — пачки чаще, ротация раскидает нагрузку


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


# --- Догенерация недостающих переводов (на все языки реестра) ---
TRANSLATE_LANGS = LANG_CODES
TRANSLATE_BATCH = int(os.getenv("TRANSLATE_BATCH", "10"))
TRANSLATE_CHECK_SEC = int(os.getenv("TRANSLATE_CHECK_SEC", "6"))




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


# --- Бэкилл буквы «ё» в русских переводах (нужно для корректной озвучки) ---
YO_BATCH = int(os.getenv("YO_BATCH", "30"))        # сколько слов пула проверяем за один проход
YO_CHECK_SEC = int(os.getenv("YO_CHECK_SEC", "6"))


async def yo_fix_loop():
    """Фоновый бэкилл «ё» в русских переводах: пачками берём непроверенные слова (yo_done=0),
    восстанавливаем «ё» где нужно и через update_pool_translate обновляем текст (озвучка перевода
    сбрасывается и регенерится сама). Идёт до конца один раз; новые слова уже получают «ё» из промпта."""
    await asyncio.sleep(40)
    while True:
        try:
            pend = await yo_pending(YO_BATCH)
            if not pend:
                await asyncio.sleep(300)   # всё проверено — ждём долго (фактически простой)
                continue
            # уникальные русские строки-кандидаты (без «е» буква «ё» точно не нужна — не шлём в LLM)
            cand = {v for _pid, data in pend
                    for v in ((data.get("translate") or {}).get("ru") or [])
                    if isinstance(v, str) and ("е" in v or "Е" in v)}
            fix = {}
            if cand:
                if not llm.text_available("autofill"):
                    logger.info("yo_fix_loop: бюджет моделей/ключей исчерпан")
                    await asyncio.sleep(300)
                    continue
                fix = await restore_yo(sorted(cand))
            changed = 0
            for pid, data in pend:
                tr = data.get("translate") or {}
                ru = tr.get("ru") or []
                new_ru = [fix.get(v, v) for v in ru]
                if new_ru != ru:
                    await update_pool_translate(pid, {**tr, "ru": new_ru})
                    changed += 1
                await mark_yo_done(pid)
            if changed:
                logger.info(f"yo_fix: восстановлено «ё» у {changed}/{len(pend)} слов (озвучка перегенерится)")
            await asyncio.sleep(YO_CHECK_SEC)
        except Exception as e:
            errors.report(e, "yo_fix_loop")
            await asyncio.sleep(60)


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


# --- Фоновый дедуп пула: слияние слов-дублей (вариантов написания одного слова) ---
DEDUP_ENABLED = os.getenv("DEDUP_ENABLED", "true").lower() == "true"
DEDUP_COS = float(os.getenv("DEDUP_COS", "0.93"))         # порог близости (cos), выше — кандидат на дубль
DEDUP_CHECK_SEC = int(os.getenv("DEDUP_CHECK_SEC", "20"))  # пауза между проверками (под RPM LLM)
_DEDUP_SYS = (
    "Ты — авторитетный лексикограф норвежского (bokmål). Тебе дают два норвежских слова с переводами. "
    "Определи отношение (relation):\n"
    "• same_word — одно и то же слово в разных написаниях (орфография/диакритика/диалект/опечатка), "
    "значение и часть речи идентичны;\n"
    "• inflected_form — одно из них является ГРАММАТИЧЕСКОЙ ФОРМОЙ другого (спряжение глагола, "
    "определённая/множественная форма сущ., склонение): напр. bærer←bære, dyret←dyr, omgivelser←omgivelse, "
    "hæren←hær. Сюда же — род. падеж и т.п.;\n"
    "• different — разные слова, синонимы, родственные но разные слова, либо нет полной уверенности.\n"
    "canonical — словарная ЛЕММА, которую нужно ОСТАВИТЬ (инфинитив / ед.ч. неопределённая / базовая "
    "форма), ровно A или B. same_word/inflected_form ставь ТОЛЬКО при полной уверенности; иначе different. "
    "Отвечай строго по схеме."
)


def _tr_brief(data):
    """Краткая строка переводов из data для промпта."""
    try:
        t = (data if isinstance(data, dict) else json.loads(data or "{}")).get("translate", {}) or {}
    except Exception:
        t = {}
    out = [f"{lng}:{t[lng][0]}" for lng in ("ru", "en", "ukr") if t.get(lng)]
    return ", ".join(out) or "—"


async def dedup_loop():
    """По одному слову из очереди (новые первыми): ищем ближайшего соседа по смыслу; если очень
    близко и LLM подтверждает, что это один и тот же слово в другом написании — сливаем (оставляем
    более используемый/каноничный вариант, второй удаляем). Флаг word_pool.dedup_done."""
    await asyncio.sleep(45)
    while True:
        if runtime.PAUSED.get("dedup"):
            await asyncio.sleep(30); continue
        try:
            pend = await dedup_pending(1)
            if not pend:
                await asyncio.sleep(300)  # очередь пуста — ждём дольше
                continue
            if not llm.text_available("autofill"):
                await asyncio.sleep(300)  # квота/ключи кончились
                continue
            pid, no, data, emb = pend[0]
            # Устойчивые выражения (pos='phrase') исключаем из дедупа: воркер заточен под СЛОВА
            # (орфо-варианты/словоформы через same_word/inflected_form), а фразы — курируемый контент
            # с дедупом ещё на генерации; семантический merge близких фраз↔фраз мог бы удалить
            # легитимно разные коллокации. Pos-гард ниже защищает фразу↔слово, этот — фразу↔фразу.
            try:
                _pp = ((json.loads(data) if isinstance(data, str) else (data or {})) or {}).get("part_of_speech", "")
            except Exception:
                _pp = ""
            if _pp == "phrase":
                await mark_dedup(pid)
                await asyncio.sleep(2)
                continue
            neighbors = await nearest_other(pid, emb, 4)
            cand = [n for n in neighbors if (1.0 - n[3]) >= DEDUP_COS]  # distance → cos
            if not cand:
                await mark_dedup(pid)
                await asyncio.sleep(2)
                continue
            nid, nno, ndata, _dist = cand[0]
            # омонимы: разные части речи = разные слова (føde «еда»/«рожать») — не сливаем
            def _pos_of(d):
                try:
                    return ((json.loads(d) if isinstance(d, str) else (d or {})) or {}).get("part_of_speech", "") or ""
                except Exception:
                    return ""
            pa, pb = _pos_of(data), _pos_of(ndata)
            if pa and pb and pa != pb:
                await mark_dedup(pid)
                await asyncio.sleep(2)
                continue
            user = (f"Слово A: «{no}» ({_tr_brief(data)})\n"
                    f"Слово B: «{nno}» ({_tr_brief(ndata)})")
            res = await ask_json(_DEDUP_SYS, user, DEDUP_SCHEMA, purpose="autofill",
                                 label=f"дедуп «{no}»≈«{nno}»")
            rel = (res or {}).get("relation")
            canon = ((res or {}).get("canonical") or "").strip()
            winner = loser = None
            if rel == "inflected_form":
                # словоформа → УДАЛЯЕМ форму, оставляем ЛЕММУ (независимо от частоты)
                if canon == no:
                    winner, loser = pid, nid
                elif canon == nno:
                    winner, loser = nid, pid
                # если канон не совпал ни с A, ни с B — не трогаем (неоднозначно)
            elif rel == "same_word":
                # орфо-вариант → оставляем более используемый/каноничный (прежняя логика)
                ua, ub = await pool_usage_count(pid), await pool_usage_count(nid)
                if ua != ub:
                    winner, loser = (pid, nid) if ua > ub else (nid, pid)
                elif canon == no:
                    winner, loser = pid, nid
                elif canon == nno:
                    winner, loser = nid, pid
                else:
                    winner, loser = (nid, pid) if nid < pid else (pid, nid)  # оставить старшее (меньший id)
            if winner:
                await merge_pool_words(winner, loser)
                await mark_dedup(winner)
                logger.info(f"dedup[{rel}]: «{no}»≈«{nno}» → оставлен id{winner}, удалён id{loser}")
            else:
                await mark_dedup(pid)
            await asyncio.sleep(DEDUP_CHECK_SEC)
        except Exception as e:
            errors.report(e, "dedup_loop")
            await asyncio.sleep(30)


# --- Частотность слов по корпусу (Zipf из статического словаря) ---
_NB_ZIPF = None


def _load_zipf():
    global _NB_ZIPF
    if _NB_ZIPF is None:
        path = os.path.join(os.path.dirname(__file__), "data", "nb_zipf.json")
        try:
            with open(path, encoding="utf-8") as f:
                _NB_ZIPF = json.load(f)
        except Exception as e:
            logger.warning(f"nb_zipf.json не загружен: {e}")
            _NB_ZIPF = {}
    return _NB_ZIPF


def _free_zipf():
    """Освободить словарь частот (≈45МБ) — держим в RAM только во время добора."""
    global _NB_ZIPF
    _NB_ZIPF = None


async def compound_index_loop():
    """Наполняет обратный индекс частей составных слов (word_pool_compounds) из ordbank: идём
    по пулу ВПЕРЁД по id (курсор в app_settings), новые слова подхватываются на следующих кругах.
    Без LLM/сети — ordbank.compound() локальный lookup. Нужен для разблокировки композитов по
    выученным основам (session/compounds + suggest_compounds)."""
    from db import get_setting, set_setting, ordbank
    from db.compound_index import pool_batch_after, set_pool_compounds
    await asyncio.sleep(30)
    while True:
        if runtime.PAUSED.get("compound_index"):
            await asyncio.sleep(20); continue
        try:
            cursor = int(await get_setting("compound_idx_cursor") or 0)
            batch = await pool_batch_after(cursor, 300)
            if not batch:
                await asyncio.sleep(6 * 3600)   # весь пул пройден — редкая проверка новых слов
                continue
            rows = [(pid, no, c["forledd"], c["etterledd"])
                    for pid, no in batch if (c := ordbank.compound(no))]
            await set_pool_compounds(rows)
            await set_setting("compound_idx_cursor", str(batch[-1][0]))
        except Exception as e:
            errors.report(e, "compound_index_loop")
            await asyncio.sleep(60)
        await asyncio.sleep(2)


async def freq_loop():
    """Проставляет частотность (Zipf 0..8) словам без неё из статического корпус-словаря.
    Без LLM/сети. Словарь грузится только когда есть работа и освобождается в простое —
    в покое RAM ≈ 0. Слово вне корпуса → 0.0 (очень редкое). Покрывает бэкафилл и новые слова."""
    await asyncio.sleep(35)
    while True:
        try:
            pend = await freq_pending(500)
            if not pend:
                _free_zipf()                 # работы нет — отдаём память
                await asyncio.sleep(120)     # подстраховка (freq ставится при вставке слова)
                continue
            z = await asyncio.to_thread(_load_zipf)   # 5.5МБ json.load — не блокируем event loop
            if not z:
                return
            pairs = [(pid, float(z.get((no or "").strip().lower(), 0.0))) for pid, no in pend]
            n = await set_pool_freq_bulk(pairs)
            if n:
                logger.info(f"freq: проставлено {n}")
            await asyncio.sleep(2)
        except Exception as e:
            errors.report(e, "freq_loop")
            await asyncio.sleep(30)


# --- Переразметка части речи для слов с пустой/нераспознанной POS («прочее») ---
POS_BATCH = int(os.getenv("POS_BATCH", "20"))
POS_CHECK_SEC = int(os.getenv("POS_CHECK_SEC", "5"))
_POS_SYS = (
    "Ты — лингвист по норвежскому (bokmål). Для каждого слова определи часть речи СТРОГО "
    "одним ключом из списка: noun (существительное), verb (глагол), adjective (прилагательное), "
    "adverb (наречие), preposition (предлог), conjunction (союз), pronoun (местоимение), "
    "determiner (детерминатив/артикль), numeral (числительное), interjection (междометие), "
    "phrase (устойчивое выражение из нескольких слов). Поле word — ровно как на входе."
)


async def pos_loop():
    """Фоновая переразметка части речи: берёт слова с пустой/нераспознанной POS, определяет
    правильную (enum) и пишет в data.part_of_speech. Идёт перед формами (форм без верной POS нет)."""
    await asyncio.sleep(25)
    while True:
        if runtime.PAUSED["pos"]:
            await asyncio.sleep(20); continue
        try:
            if llm.text_enabled() and llm.text_available("autofill"):
                rows = await pos_uncategorized(POS_BATCH)
                if rows:
                    user = "Слова:\n" + "\n".join(f"- {nw}" for _, nw, _ in rows)
                    try:
                        data = await ask_json(_POS_SYS, user, POS_REFINE_SCHEMA, purpose="autofill",
                                              label=f"часть речи ({len(rows)})", temperature=0)
                    except Exception as e:
                        errors.report(e, "pos_loop")
                        data = None
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
                        pos = r.get("part_of_speech")
                        if not pid or pid in seen or pos not in POS_KEYS:
                            continue
                        seen.add(pid)
                        await set_pool_pos(pid, pos)
                        done += 1
                    logger.info(f"pos_loop: размечено {done}/{len(rows)}")
        except Exception as e:
            errors.report(e, "pos_loop")
        await asyncio.sleep(POS_CHECK_SEC)


# --- Грамматические формы по части речи (фоновая очередь, отдельный промпт на POS) ---
FORMS_BATCH = int(os.getenv("FORMS_BATCH", "20"))
FORMS_CHECK_SEC = int(os.getenv("FORMS_CHECK_SEC", "5"))




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
                    # формы — сначала детерминированно: дамп ordbank → живой ordbøkene
                    # (nyord, кэш в ordbank_ext) → и только остаток генерит LLM
                    from db import ordbank
                    await ordbank.ensure_file()
                    left, det = [], 0
                    for pid, nw, data in rows:
                        f = ordbank.lookup(nw, cat) or await ordbank.lookup_online(nw, cat)
                        if f:
                            await set_pool_forms(pid, f)
                            det += 1
                        else:
                            left.append((pid, nw, data))
                    if det:
                        logger.info(f"forms_loop[{cat}]: ordbank +{det}")
                    if left:
                        done = await forms_batch(cat, left)
                        logger.info(f"forms_loop[{cat}]: LLM +{done}/{len(left)}")
        except Exception as e:
            errors.report(e, "forms_loop")
        await asyncio.sleep(FORMS_CHECK_SEC)




_COUNTABLE_SYS = (
    "Ты — эксперт по норвежскому (bokmål). Для каждого существительного скажи, ИСЧИСЛЯЕМО ли оно "
    "в бытовой речи: можно ли естественно сказать «mange/flere <слово>» или «to <слово>». "
    "Неисчисляемые (масс-нуны: vann, luft, snø, informasjon, bruk, mat, kaffe) — countable=false. "
    "Обычные предметные (bil, hus, jente) — true. Поле word — ровно как на входе."
)


async def countability_loop():
    """Бэкфилл исчисляемости нунов (ОДИН проход по базе): у слов с формами, но без отметки
    uncountable, спрашиваем LLM пачками и дописываем флаг в forms. Новые слова получают отметку
    сразу в forms_batch (countable в NOUN_FORMS_SCHEMA) — очередь исчерпается и луп заснёт."""
    await asyncio.sleep(45)
    while True:
        if runtime.PAUSED.get("countability"):   # свой ключ: пауза forms больше не тормозит исчисляемость
            await asyncio.sleep(20); continue
        try:
            if llm.text_enabled() and llm.text_available("autofill"):
                rows = await nouns_missing_countability(50)
                if not rows:
                    await asyncio.sleep(6 * 3600)   # проход завершён — редкая проверка на новые хвосты
                    continue
                user = "Слова:\n" + "\n".join(f"- {nw}" for _, nw in rows)
                data = await ask_json(_COUNTABLE_SYS, user, COUNTABLE_SCHEMA, purpose="autofill",
                                      label=f"исчисляемость нунов ({len(rows)})", temperature=0)
                results = (data or {}).get("results", []) if isinstance(data, dict) else []
                by_word = {}
                for r in results:
                    if isinstance(r, dict) and isinstance(r.get("countable"), bool) and r.get("word"):
                        by_word[str(r["word"]).strip().lower()] = r["countable"]
                positional = len(results) == len(rows)
                done = 0
                for i, (pid, nw) in enumerate(rows):
                    c = by_word.get(str(nw).strip().lower())
                    if c is None and positional and isinstance(results[i], dict) and isinstance(results[i].get("countable"), bool):
                        c = results[i]["countable"]
                    if c is None:
                        c = True   # не ответила — считаем исчисляемым (безопасный дефолт) и выводим из очереди
                    await merge_pool_forms(pid, {"uncountable": not c})
                    done += 1
                logger.info(f"countability_loop: +{done}/{len(rows)}")
        except Exception as e:
            errors.report(e, "countability_loop")
        await asyncio.sleep(FORMS_CHECK_SEC)


async def autofill_loop():
    await asyncio.sleep(15)
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
                    # два независимых бэкфила по своим типам: озвучка по СЛОВУ (bare strings из
                    # pool_missing_tts), эмбеддинги строго по ID ((pid,word) из pool_missing_embedding).
                    # Озвучка бесплатна (edge) и от квот не зависит; эмбеддинги при выжатой квоте
                    # ловят 429 — их отдельность не морит озвучку голодом.
                    m = await tts_backfill(tts_miss) if tts_miss else 0
                    n = await embed_backfill(emb_miss) if emb_miss else 0
                    logger.info(f"autofill: backfill tts {len(tts_miss)}(+{m}) emb {len(emb_miss)}(+{n})")
                elif unclassified and not runtime.PAUSED["classify"]:
                    done = await classify_batch(unclassified)  # пачка: уровень + темы
                    logger.info(f"autofill: classified {done}/{len(unclassified)}")
        except Exception as e:
            errors.report(e, "autofill_loop")
            await asyncio.sleep(120)
            continue
        await asyncio.sleep(interval)
