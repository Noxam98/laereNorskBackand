"""Пакетное LLM-обогащение слов пула: классификация (уровень/темы), описания, переводы,
грамматические формы. Зависит от db/llm/langs; реэкспортируется в autofill для воркеров.
"""
import errors
from llm import (
    ask_json, CLASSIFY_SCHEMA, DESCRIBE_BATCH_SCHEMA, TRANSLATE_BATCH_SCHEMA,
    NOUN_FORMS_SCHEMA, VERB_FORMS_SCHEMA, ADJ_FORMS_SCHEMA, CEFR_LEVELS, TOPIC_KEYS, TOPIC_TAGS,
)
from langs import LANG_CODES, LANGS_CSV
from db import get_pool_id, set_pool_description, set_pool_forms, set_pool_meta, normalize_word


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
    norm_to_item = {normalize_word(it["word"]): it for it in items}
    positional = len(results) == len(items)
    done, seen = 0, set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        rw = normalize_word(r.get("word") or "")
        # омонимы: пишем по id из выборки, а не через get_pool_id без pos (см. describe_batch)
        it = items[i] if (positional and rw == normalize_word(items[i]["word"])) else None
        if it is None:
            it = norm_to_item.get(rw)
        if it is None and positional:
            it = items[i]
        if not it:
            continue
        pid = it.get("id") or await get_pool_id(it["word"], it.get("pos"))
        if not pid or pid in seen:
            continue
        seen.add(pid)
        level = r.get("level") if r.get("level") in CEFR_LEVELS else None
        topics = [t for t in (r.get("topics") or []) if t in TOPIC_KEYS]
        if level or topics:
            await set_pool_meta(pid, level=level, topics=topics)
            done += 1
    return done
_DESCRIBE_SYS = (
    "Ты — преподаватель норвежского. Для каждого норвежского слова дай краткое "
    f"(1-2 предложения) понятное описание-толкование на каждом языке: {LANGS_CSV}. "
    "У слов в скобках указаны часть речи и переводы — описывай ИМЕННО это значение слова, "
    "а не другой смысл того же написания. "
    "Верни results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)
async def describe_batch(words):
    """Пакетные описания (≤DESCRIBE_BATCH слов за один LLM-вызов). Кешируется в БД.
    words — [{word, pos, translate}] (или строки для совместимости)."""
    if not words:
        return 0
    items = [({"word": w} if isinstance(w, str) else dict(w)) for w in words]
    lines = []
    for it in items:
        ctx = []
        if it.get("pos"):
            ctx.append(f"часть речи: {it['pos']}")
        tr = it.get("translate") or {}
        for l in LANG_CODES:
            vals = [v for v in (tr.get(l) or []) if v]
            if vals:
                ctx.append(f"{l}: {', '.join(vals)}")
        lines.append(f"- {it['word']}" + (f" ({'; '.join(ctx)})" if ctx else ""))
    user = "Опиши слова:\n" + "\n".join(lines)
    try:
        data = await ask_json(_DESCRIBE_SYS, user, DESCRIBE_BATCH_SCHEMA, purpose="autofill",
                              label=f"описания слов ({len(items)})")
    except Exception as e:
        errors.report(e, "describe_batch")
        return 0
    results = (data or {}).get("results", []) if isinstance(data, dict) else []
    # Сохраняем описание по ВХОДНОМУ слову (что спрашивали), а не по тому, как модель его
    # переписала: по нормализованному имени, иначе позиционно. Иначе слова с «кривым»
    # написанием (модель его «исправляет») никогда не сохраняются и крутят очередь вечно.
    norm_to_item = {normalize_word(it["word"]): it for it in items}
    positional = len(results) == len(items)
    done, seen = 0, set()
    for i, r in enumerate(results):
        if not isinstance(r, dict):
            continue
        rw = normalize_word(r.get("word") or "")
        # точное позиционное совпадение различает омонимы (одно написание, разный pos) в пачке;
        # иначе — по нормализованному слову; иначе позиционно (модель «исправила» написание).
        it = items[i] if (positional and rw == normalize_word(items[i]["word"])) else None
        if it is None:
            it = norm_to_item.get(rw)
        if it is None and positional:
            it = items[i]
        if not it:
            continue
        # пишем строго по id из выборки: get_pool_id без pos попал бы в старшую запись омонима,
        # description у нужной записи остался бы NULL — и очередь крутила бы её вечно (слив квоты).
        pid = it.get("id") or await get_pool_id(it["word"], it.get("pos"))
        if not pid or pid in seen:
            continue
        seen.add(pid)
        desc = {k: (r.get(k) or "") for k in LANG_CODES}
        await set_pool_description(pid, desc)
        done += 1
    return done
_TRANSLATE_SYS = (
    "Ты — переводчик с норвежского (bokmål). Для каждого норвежского слова дай перевод "
    f"на {len(LANG_CODES)} языков: {LANGS_CSV} — по 1-3 варианта (массив строк), без пояснений. "
    "Пиши орфографически точно для синтеза речи: в русском используй букву «ё» там, где она нужна "
    "(а не «е»: мёд, ёлка, объём, всё, актёр), в других языках — корректную диакритику. "
    "Верни results: по объекту на каждое входное слово (поле word — ровно как на входе)."
)
async def translate_batch(words):
    """Пакетный перевод списка норвежских слов на 6 языков. Возвращает {norm_word: result}."""
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
_FORMS = {
    "noun": dict(
        schema=NOUN_FORMS_SCHEMA, fields=["gender", "def_sg", "indef_pl", "def_pl"],
        sys=("Ты — эксперт по существительным норвежского (bokmål). Для каждого слова дай ТОЧНЫЕ "
             "формы (включая нерегулярные). gender — ОБЯЗАТЕЛЬНО один из en/ei/et, НИКОГДА не пусто "
             "(женский род давай 'ei'; примеры: en gutt, ei jente, et hus). def_sg (опр. ед.), "
             "indef_pl (неопр. мн.), def_pl (опр. мн.). countable — можно ли естественно сказать "
             "«mange <слово>» в бытовой речи: у неисчисляемых (vann, luft, informasjon, bruk, snø) "
             "false, у обычных предметных true. Поле word — ровно как на входе.")),
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
# Анти-залип noun-без-gender: счётчик промахов рода по pid (в памяти процесса). gender — источник
# артикля (en/ei/et), поэтому ретраим, надеясь, что модель его вернёт; но после CAP промахов сдаёмся
# и сохраняем слово без рода — чтобы оно вышло из очереди pos_missing_forms и не сливало квоту Gemini
# вечными перезапросами (хвост misclassified-слов, которым род не присвоить). Раз сохранили — слово
# больше не в очереди, так что рестарт процесса (сброс счётчика) повторно его не тянет.
_NOUN_NOGENDER_TRIES = {}
_NOUN_NOGENDER_CAP = 3


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
        # исчисляемость нуна: неисчисляемые (countable=false) не дриллятся по мн.ч. в треке форм
        if category == "noun" and isinstance(r.get("countable"), bool):
            forms["uncountable"] = not r["countable"]
        # Анти-залип noun без gender: ретраим ОГРАНИЧЕННО (gender = источник артикля en/ei/et —
        # ждём, что модель вернёт его на след. круге; enum+required в схеме делают промах редким).
        # После CAP промахов сдаёмся и сохраняем без рода, иначе misclassified-слово сливало бы квоту
        # вечно (см. константы выше).
        if category == "noun" and not forms.get("gender"):
            if _NOUN_NOGENDER_TRIES.get(pid, 0) + 1 < _NOUN_NOGENDER_CAP:
                _NOUN_NOGENDER_TRIES[pid] = _NOUN_NOGENDER_TRIES.get(pid, 0) + 1
                continue
        _NOUN_NOGENDER_TRIES.pop(pid, None)   # успех или сдача — счётчик больше не нужен
        # сохраняем даже пустые формы (только pos) — чтобы слово считалось «обработанным» и не зациклилось
        await set_pool_forms(pid, {"pos": category, **forms})
        done += 1
    return done
