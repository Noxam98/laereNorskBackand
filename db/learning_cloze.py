"""Cloze для служебных слов (A1, Ф4): генерация и кэш «вставь пропущенное». Вынесено из learning.py.

1-е предложение — ЯКОРЬ из выверенного example.no (гарантированно осмысленно), остальные — динамика
gemini-3.5-flash из ВЫУЧЕННЫХ слов юзера (по убыванию частотности). Публичное ядру: get_cloze_map
(сборка сессии), generate_cloze (фон из apply_result), _blank_example (нужен ещё exams.py) —
реэкспортируются обратно в learning (см. конец learning.py)."""
import json

from .core import _conn, _release, _now
from .learning import _fetch_user_words, status_of, is_function_word

CLOZE_N = 3
_CLOZE_SCHEMA = {"name": "cloze", "schema": {"type": "object", "properties": {"items": {"type": "array", "items": {
    "type": "object", "properties": {
        "blank": {"type": "string"}, "answer": {"type": "string"},
        "used": {"type": "array", "items": {"type": "string"}}},
    "required": ["blank", "answer", "used"]}}}, "required": ["items"]}}
_CLOZE_SYS = ("Du er norsklærer på nivå A1. Lag 3 FORSKJELLIGE, korte (≤7 ord), enkle og MENINGSFULLE "
              "setninger på naturlig bokmål, der hver bruker målordet riktig. Bruk ellers bare ord fra "
              "lista over kjente ord (bøyning er lov), men VELG ordene slik at setningen faktisk GIR MENING "
              "— ikke sett sammen tilfeldige ord (f.eks. «Jeg arbeider istedenfor en brann» er FORBUDT, "
              "meningsløst). Hver setning skal være noe en nordmann faktisk kan si og MÅ være KOMPLETT — "
              "ikke avslutt med målordet hvis det trenger en fortsettelse (f.eks. «istedenfor X» — ta med X; "
              "følg samme struktur som mønster-eksempelet). For hver: blank "
              "(setningen med ___ i stedet for målordet), answer (=målordet), used (grunnformene du brukte, "
              "utenom målordet). Korrekt grammatikk og ordstilling. Kun JSON etter skjema.")


async def get_cloze_map(db, user_id, pool_ids):
    """Кэш cloze для набора слов → {pool_id: [items]}; items=[{blank, answer, options}]."""
    if not pool_ids:
        return {}
    marks = ",".join("?" for _ in pool_ids)
    out = {}
    async with db.execute(f"SELECT pool_id, data FROM cloze_cache WHERE user_id=? AND pool_id IN ({marks})",
                          [user_id, *pool_ids]) as cur:
        for r in await cur.fetchall():
            try: out[r["pool_id"]] = json.loads(r["data"])
            except Exception: pass
    return out


async def _mastered_words(db, user_id):
    """Норвежские формы ВЫУЧЕННЫХ (mastered) слов юзера, ПО УБЫВАНИЮ ЧАСТОТНОСТИ (самые
    употребимые/простые — первыми) — допустимый словарь для cloze-предложений. Частотный
    порядок важен: алфавитный (как было) давал биас на редкие слова на «a…» (ansatte, arbeider)
    и из них модель строила неестественные предложения."""
    rows = await _fetch_user_words(db, user_id)
    out = []
    for r in rows:
        try: modes = json.loads(r.get("modes") or "{}")
        except Exception: modes = {}
        if status_of(r, modes) == "mastered":
            out.append((r["norwegian"], r.get("freq")))
    out.sort(key=lambda x: (x[1] is None, -(x[1] or 0)))   # freq DESC; None — в хвост
    return [no for (no, _f) in out]


def _blank_example(sentence, target):
    """Превратить выверенный пример в cloze: заменить целевое слово на ___ (по границе слова,
    регистронезависимо). Возвращает строку с ___ или None, если целевого слова в примере нет."""
    import re
    s = (sentence or "").strip()
    if not s or not target:
        return None
    pat = re.compile(r"\b" + re.escape(target) + r"\b", re.IGNORECASE)
    return pat.sub("___", s, count=1) if pat.search(s) else None


async def generate_cloze(user_id, pool_id):
    """Сгенерировать и закэшировать CLOZE_N cloze-предложений для служебного слова. 1-е — ЯКОРЬ из
    выверенного example.no (гарантированно осмысленное), остальные — динамика 3.5-flash из ВЫУЧЕННЫХ
    слов юзера (по убыванию частотности). Лениво/в фоне. Перебор всех ключей-аккаунтов на 429."""
    import random
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian, data FROM word_pool WHERE id=?", (pool_id,)) as cur:
            w = await cur.fetchone()
        if not w:
            return None
        target = w["norwegian"]
        try: wdata = json.loads(w["data"]) if w["data"] else {}
        except Exception: wdata = {}
        pos = (wdata.get("part_of_speech") or "").strip().lower()
        _ex = wdata.get("example") or {}
        ex_no = (_ex.get("no") or "").strip() if isinstance(_ex, dict) else ""
        allowed = await _mastered_words(db, user_id)
        if len(allowed) < 4:
            return None
        # дистракторы той же POS — SQL-фильтр по data (без парса всех 6000 строк на слабом CPU)
        distractors = []
        if pos:
            async with db.execute(
                "SELECT norwegian, data FROM word_pool WHERE id != ? AND data LIKE ?",
                (pool_id, f'%"{pos}"%')) as cur:
                for rr in await cur.fetchall():
                    try: dd = json.loads(rr["data"]) if rr["data"] else {}
                    except Exception: dd = {}
                    if (dd.get("part_of_speech") or "").strip().lower() == pos and is_function_word(rr["norwegian"], dd):
                        distractors.append(rr["norwegian"])
    finally:
        await _release(db)
    random.shuffle(distractors)
    allowed_s = ", ".join(list(dict.fromkeys(allowed))[:60])   # топ-60 частотных (порядок уже по freq)

    def _opts():
        o = [target] + distractors[:3]
        random.shuffle(o)
        return o

    # Динамика: 3.5-flash (reasoning) даёт ОСМЫСЛЕННЫЕ предложения (lite лепил словесный салат);
    # reasoning_effort="low" + запас max_tokens — иначе «размышления» обрезают JSON. ПЕРЕБИРАЕМ ВСЕ
    # ключи-аккаунты: на 429 одного ключа (суточный лимит ~20 на 3.5-flash) — пробуем следующий.
    from llm.client import get_client
    from llm.settings import LLM_API_KEYS
    client = get_client()
    pattern = f" Riktig mønster (følg samme struktur): «{ex_no}»." if ex_no else ""
    msgs = [{"role": "system", "content": _CLOZE_SYS},
            {"role": "user", "content": f"Målord: «{target}» ({pos}).{pattern}\nKjente ord: {allowed_s}"}]
    raw = []
    for key in (LLM_API_KEYS or [""]):
        try:
            c = client.with_options(api_key=key or "not-needed", max_retries=0, timeout=60)
            resp = await c.chat.completions.create(
                model="gemini-3.5-flash", messages=msgs,
                response_format={"type": "json_schema", "json_schema": _CLOZE_SCHEMA},
                reasoning_effort="low", max_tokens=3000)
            content = resp.choices[0].message.content if (resp and resp.choices) else None
            cand = (json.loads(content).get("items") if content else []) or []
            if cand:
                raw = cand
                break
        except Exception:
            continue   # 429/таймаут/обрыв JSON — следующий ключ

    items = []
    # ЯКОРЬ: 1-е cloze из выверенного примера карточки (если в нём есть целевое слово) —
    # гарантированно осмысленно, лечит «болтающиеся» трудные слова (istedenfor и т.п.).
    anchor = _blank_example(ex_no, target)
    if anchor:
        items.append({"blank": anchor, "answer": target, "options": _opts()})
    # динамика добивает до CLOZE_N (без повтора якоря)
    for it in raw:
        if len(items) >= CLOZE_N:
            break
        blank = (it.get("blank") or "").strip()
        if "___" not in blank or any(blank == x["blank"] for x in items):
            continue
        items.append({"blank": blank, "answer": target, "options": _opts()})
    if not items:
        return None
    db = await _conn()
    try:
        await db.execute("INSERT OR REPLACE INTO cloze_cache (user_id, pool_id, data, created_at) VALUES (?,?,?,?)",
                         (user_id, pool_id, json.dumps(items, ensure_ascii=False), _now()))
        await db.commit()
    finally:
        await _release(db)
    return items
