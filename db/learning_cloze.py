"""Cloze для служебных слов (A1-A2, Ф4): «вставь пропущенное». Вынесено из learning.py.

Предложения и дистракторы берутся из СТАТИЧЕСКОГО банка (db.cloze_bank ← data/cloze-bank.json),
выверенного офлайн, а НЕ генерятся на лету LLM из выученных слов юзера. Почему отказались от
пер-юзерной генерации (была gemini-3.5-flash):
  - дистракторы-соседи по эмбеддингу давали СИНОНИМЫ (баг «samt»: 4 варианта-синонима);
  - генерация жгла квоту Gemini на каждого юзера и часто лепила неестественные предложения.
Банк даёт грамматически-контрастные дистракторы (служебные другого типа связи) и всегда
осмысленные A1-A2 предложения из фиксированного ядра. См. db/cloze_bank.py.

Публичное ядру: get_cloze_map (сборка сессии), _blank_example (нужен ещё exams.py) —
реэкспортируются обратно в learning (см. конец learning.py)."""
import random

from . import cloze_bank

CLOZE_N = 3   # клеток cloze в рампе (srs.cells.FUNC_CELLS) — до стольких items отдаём на слово


def _with_options(it):
    """Item банка {blank, answer, distractors[, tr, answer_tr, distractor_tr]} → item сессии
    {blank, answer, options[, optionsTr, sentTr]}. options — перемешанные answer + до 3 дистракторов
    («выбор из 4»). Если в банке есть переводы (7 языков), доносим их для показа НА РАЗБОРЕ:
    optionsTr[слово] = {lang: значение} (ответ — контекстно, дистракторы — общее значение),
    sentTr = {lang: перевод всего предложения}. Нет переводов в банке → поля не добавляем (фронт
    их игнорирует) — код совместим со старым банком без tr."""
    ans = it["answer"]
    ds = (it.get("distractors") or [])[:3]
    opts = [ans, *ds]
    random.shuffle(opts)
    out = {"blank": it["blank"], "answer": ans, "options": opts}
    tr_by_word = {}
    if it.get("answer_tr"):
        tr_by_word[ans] = it["answer_tr"]
    dtr = it.get("distractor_tr") or []
    for i, d in enumerate(ds):
        if i < len(dtr) and dtr[i]:
            tr_by_word[d] = dtr[i]
    if tr_by_word:
        out["optionsTr"] = tr_by_word
    if it.get("tr"):
        out["sentTr"] = it["tr"]
    return out


async def get_cloze_map(db, user_id, pool_ids):
    """Cloze для набора служебных слов → {pool_id: [items]}; item = {blank, answer, options}.
    Источник — статический банк по (norwegian, pos). Слова без записи в банке пропускаем (их
    рампа — FUNC_CHOICE, до cloze они не доходят). Список добиваем до CLOZE_N цикличным повтором
    (если валидация оставила <3 предложений) — каждый повтор с новым перемешиванием options."""
    if not pool_ids:
        return {}
    import json
    marks = ",".join("?" for _ in pool_ids)
    out = {}
    async with db.execute(
            f"SELECT id, norwegian, data FROM word_pool WHERE id IN ({marks})", list(pool_ids)) as cur:
        rows = await cur.fetchall()
    for r in rows:
        try:
            d = json.loads(r["data"]) if r["data"] else {}
        except Exception:
            d = {}
        raw = cloze_bank.items_for(r["norwegian"], d.get("part_of_speech"))
        if not raw:
            continue
        built = [_with_options(it) for it in raw[:CLOZE_N]]
        while len(built) < CLOZE_N:                     # <3 выверенных — доцикливаем с новым shuffle
            built.append(_with_options(raw[len(built) % len(raw)]))
        out[r["id"]] = built
    return out


def _blank_example(sentence, target):
    """Превратить выверенный пример в cloze: заменить целевое слово на ___ (по границе слова,
    регистронезависимо). Возвращает строку с ___ или None, если целевого слова в примере нет.
    Используется exams.py для cloze-вопросов служебных из их example.no."""
    import re
    s = (sentence or "").strip()
    if not s or not target:
        return None
    pat = re.compile(r"\b" + re.escape(target) + r"\b", re.IGNORECASE)
    return pat.sub("___", s, count=1) if pat.search(s) else None
