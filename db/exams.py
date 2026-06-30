"""Зачётный экзамен-ворота (§2.4-A) + аудит-экзамен забывания (§2.4-B).
Пейсинг притока новых слов и проверка забывания сертифицированных. Вынесено из learning.py.
ВАЖНО: _is_certified и _audit_throttled нужны ядру (build_session/get_due/learning_stats) —
они определены здесь и реэкспортируются обратно в learning (см. конец learning.py).
"""
import json
from .core import _conn, _release, _now
from .learning import (
    ALL_CELLS, AUDIT_CAP, FIRST_AUDIT_DAYS, PACK, PACK_FIRST, PASS, REQUIRED_CELLS, SAMPLE,
    THROTTLE, THROTTLE_DAYS, _AUDIT_GROWTH, _blank_example, _due_str,
    _fetch_user_words, is_function_word, required_cells, status_of, fuzzy,
)
# _KNOWN_VOCAB — модульный кэш известной лексики, определён НИЖЕ в этом же модуле (не импортируем)


# ---------------- зачётный экзамен-ворота (§2.4-A) ----------------

def _is_certified(r):
    return bool(r.get("certified"))


def _pack_rows(rows):
    """«Несданная пачка» = выученные (mastered), ещё не сертифицированные слова."""
    out = []
    for r in rows:
        try:
            modes = json.loads(r.get("modes") or "{}")
        except Exception:
            modes = {}
        if status_of(r, modes) == "mastered" and not _is_certified(r):
            out.append(r)
    return out


async def _had_certification(db, user_id):
    """Была ли уже хоть одна сертификация (для выбора порога: первый — PACK_FIRST)."""
    async with db.execute(
        "SELECT 1 FROM user_words WHERE user_id = ? AND certified = 1 LIMIT 1", (user_id,)) as cur:
        return (await cur.fetchone()) is not None


async def gate_status(user_id):
    """Состояние ворот: {pack, threshold, open}.
    pack — размер несданной пачки; threshold — PACK_FIRST до первой сертификации, дальше PACK;
    open — ворота открыты на сдачу (pack достиг порога)."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        had = await _had_certification(db, user_id)
    finally:
        await _release(db)
    pack = len(_pack_rows(rows))
    threshold = PACK if had else PACK_FIRST
    return {"pack": pack, "threshold": threshold, "open": pack >= threshold}


async def _audit_throttled(db, user_id):
    """Активен ли мягкий тормоз аудита (users.audit_throttle_until в будущем)."""
    async with db.execute("SELECT audit_throttle_until FROM users WHERE id = ?", (user_id,)) as cur:
        row = await cur.fetchone()
    until = row["audit_throttle_until"] if row else None
    return bool(until) and until > _now()


async def audit_throttled(user_id):
    """Снаружи: активен ли мягкий тормоз притока новых после аудита (>THROTTLE забытого)."""
    db = await _conn()
    try:
        return await _audit_throttled(db, user_id)
    finally:
        await _release(db)


async def new_words_blocked(user_id):
    """Приток новых слов закрыт, если ворота ждут сдачи (§2.4-A) ИЛИ активен мягкий тормоз
    аудита (§2.4-B: после >THROTTLE забытого приток новых притормаживается на THROTTLE_DAYS)."""
    if (await gate_status(user_id))["open"]:
        return True
    db = await _conn()
    try:
        return await _audit_throttled(db, user_id)
    finally:
        await _release(db)


def _exam_pools(rows, lang):
    """Пулы дистракторов: переводы (no2int), норвежские слова (int2no), норвежские служебные (cloze)."""
    tr_pool, no_pool, func_pool = [], [], []
    for r in rows:
        data = json.loads(r["data"]) if r.get("data") else {}
        t = (data.get("translate", {}).get(lang) or [None])[0]
        if t:
            tr_pool.append(t)
        no = r["norwegian"]
        if no:
            no_pool.append(no)
            if is_function_word(no, data):
                func_pool.append(no)
    return tr_pool, no_pool, func_pool


def _pick_distract(pool, exclude, n=3):
    """n уникальных дистракторов из пула (в случайном порядке), исключая exclude (регистр игнор)."""
    import random
    seen = {x.strip().lower() for x in exclude}
    out = []
    for w in (random.sample(pool, len(pool)) if pool else []):
        lw = (w or "").strip().lower()
        if not lw or lw in seen:
            continue
        seen.add(lw); out.append(w)
        if len(out) >= n:
            break
    return out


def _exam_question(r, lang, tr_pool, no_pool, func_pool):
    """Типизированный вопрос: cloze (служебные с примером) | int2no | no2int — все «выбор из 4».
    Ответ сверяется типонезависимо (_exam_answer_ok: перевод ИЛИ норвежское слово)."""
    import random
    data = json.loads(r["data"]) if r.get("data") else {}
    no = r["norwegian"]
    tr = data.get("translate", {}).get(lang) or []
    corr_tr = tr[0] if tr else None
    ex = data.get("example") or {}
    ex_no = (ex.get("no") or "").strip() if isinstance(ex, dict) else ""

    # cloze — для служебных слов с примером (пропуск + варианты-служебные). Ответ (no) НЕ шлём
    # клиенту — иначе видно правильный вариант; грейд по pool_id на сервере.
    if is_function_word(no, data) and ex_no:
        blank = _blank_example(ex_no, no)
        distract = _pick_distract(func_pool, {no}, 3)
        if blank and len(distract) == 3:
            opts = distract + [no]; random.shuffle(opts)
            return {"type": "cloze", "blank": blank, "pool_id": r["pool_id"], "options": opts}

    if not corr_tr:
        return None

    roll = random.random()
    # ввод текста (~25%): печатаешь норвежское по переводу, без вариантов; грейд нечёткий (fuzzy)
    if roll < 0.25:
        return {"type": "input", "prompt": corr_tr, "pool_id": r["pool_id"]}
    # int2no (~35%): перевод → выбрать норвежское (ответ no — среди options, отдельно не шлём)
    if roll < 0.6:
        distract = _pick_distract(no_pool, {no}, 3)
        if len(distract) == 3:
            opts = distract + [no]; random.shuffle(opts)
            return {"type": "int2no", "prompt": corr_tr, "pool_id": r["pool_id"], "options": opts}
    # no2int: норв. слово (видимый вопрос) → выбрать перевод
    distract = _pick_distract(tr_pool, {corr_tr}, 3)
    if len(distract) < 3:
        return None
    opts = distract + [corr_tr]; random.shuffle(opts)
    return {"type": "no2int", "no": no, "pool_id": r["pool_id"], "options": opts}


_KNOWN_VOCAB = None


async def _known_vocab(db):
    """Множество нормализованных норв. слов пула — словарь-страж для нечёткой сверки (ввод, совпавший
    с ДРУГИМ известным словом, не считаем опечаткой). Кэш на процесс (новые слова редки)."""
    global _KNOWN_VOCAB
    if _KNOWN_VOCAB is None:
        async with db.execute("SELECT norwegian FROM word_pool") as cur:
            _KNOWN_VOCAB = {fuzzy.normalize(r["norwegian"]) for r in await cur.fetchall() if r["norwegian"]}
    return _KNOWN_VOCAB


def _exam_answer_ok(r, lang, ans, known=None):
    """Нечёткая типонезависимая сверка (fuzzy.py): ответ достаточно близок к любой «своей» форме —
    перевод(ы) на lang, норвежское слово ИЛИ его словоформы. Со словарём-стражем (known) ввод,
    равный ДРУГОМУ известному слову, отклоняется (это не опечатка)."""
    data = json.loads(r["data"]) if r.get("data") else {}
    try:
        fcol = json.loads(r["forms"]) if r.get("forms") else None
    except Exception:
        fcol = None
    acc = list(data.get("translate", {}).get(lang) or [])
    acc += fuzzy.word_forms(r["norwegian"], fcol)
    return fuzzy.fuzzy_match(ans, acc, known=known)


async def build_gate_exam(user_id, lang="ru"):
    """До SAMPLE случайных вопросов из несданной пачки.
    Формат вопроса как в build_placement: {no, pool_id, options:[4]}."""
    import random
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
    finally:
        await _release(db)
    pack = _pack_rows(rows)
    tr_pool, no_pool, func_pool = _exam_pools(rows, lang)
    random.shuffle(pack)
    questions = []
    for r in pack:
        if len(questions) >= SAMPLE:
            break
        q = _exam_question(r, lang, tr_pool, no_pool, func_pool)
        if q:
            questions.append(q)
    return {"questions": questions, "sample": SAMPLE, "pass": PASS}


def _demote_fields(modes, cells=REQUIRED_CELLS):
    """Демоут слова (забыл на аудите / провал зачёта): слово возвращается в очередь СРАЗУ на
    ПОСЛЕДНЕЙ ступени рампы (ввод / последний cloze), а не гоняется заново с выбора. Все ступени,
    кроме последней, помечаем пройденными ('1'), последнюю — '' (pending → её и вернёт _next_step).
    Если на ней не вспомнит — apply_result откатит на ступень назад (build) и доучит обычной рампой.
    Силу/историю/ease — вниз, certified=0. Возвращает (modes_json, strength, ease)."""
    m = {k: v for k, v in (modes or {}).items() if k not in ALL_CELLS and k != "hist"}
    m["hist"] = ""
    for c in ALL_CELLS:
        m[c] = ""
    for c in cells[:-1]:        # все ступени, кроме последней, — пройдены
        m[c] = "1"
    return json.dumps(m, ensure_ascii=False), 0, 1.3


async def _demote(db, user_id, r):
    """Перевести слово mastered→review: сброс клеток рампы, силы/ease вниз, certified=0,
    вернуть в ближайшую ротацию (due завтра, lapses+1, reps→review-уровень)."""
    try:
        modes = json.loads(r.get("modes") or "{}")
    except Exception:
        modes = {}
    modes_json, strength, ease = _demote_fields(modes, required_cells(r))
    await db.execute("""
        UPDATE user_words
        SET modes = ?, strength = ?, ease = ?, certified = 0, mastered = 0,
            reps = MAX(1, reps), lapses = lapses + 1, interval_days = 1, streak = 0,
            due_at = ?, last_seen = ?
        WHERE user_id = ? AND pool_id = ?
    """, (modes_json, strength, ease, _due_str(1), _now(), user_id, r["pool_id"]))


async def grade_gate_exam(user_id, answers, lang="ru"):
    """Оценить зачётный экзамен. answers: [{pool_id, answer}].
    Верных ≥ PASS → сертифицировать всю текущую несданную пачку (certified=1), {passed:True}.
    Иначе провал: демоут каждого промаха (mastered→review) + столько же самых слабых слов
    пачки по strength (штраф ×2 промаха). {passed:False, demoted:N}."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        pack = _pack_rows(rows)
        by_pid = {r["pool_id"]: r for r in pack}
        known = await _known_vocab(db)
        # сверка ответов — нечётко, по любой «своей» форме слова (перевод/норв./словоформы)
        correct_n = 0
        missed_pids = []
        seen = set()   # дедуп по pool_id: одна карточка засчитывается ОДИН раз. Иначе повтор верного
        for a in (answers or []):   # ответа ×PASS подделывал бы сдачу ворот (сертификацию всей пачки 50–100 слов)
            pid = a.get("pool_id")
            r = by_pid.get(pid)
            if not r or pid in seen:
                continue
            seen.add(pid)
            if _exam_answer_ok(r, lang, a.get("answer"), known):
                correct_n += 1
            else:
                missed_pids.append(pid)

        if correct_n >= PASS:
            if pack:
                marks = ",".join("?" for _ in pack)
                # сертифицируем пачку и сразу назначаем первый аудит забывания (now + FIRST_AUDIT_DAYS).
                # audit_interval запоминает длину текущего интервала — при успехе аудита он умножается на рост.
                await db.execute(
                    f"UPDATE user_words SET certified = 1, was_certified = 1, audit_due = ?, audit_interval = ? "
                    f"WHERE user_id = ? AND certified = 0 AND pool_id IN ({marks})",
                    [_due_str(FIRST_AUDIT_DAYS), float(FIRST_AUDIT_DAYS), user_id]
                    + [r["pool_id"] for r in pack])
                await db.commit()
            return {"passed": True}

        # провал: демоут промахов + столько же самых слабых по силе из пачки (штраф ×2)
        missed = [by_pid[p] for p in missed_pids if p in by_pid]
        missed_set = {r["pool_id"] for r in missed}
        rest = sorted([r for r in pack if r["pool_id"] not in missed_set],
                      key=lambda r: (r.get("strength") or 0, r.get("due_at") or ""))
        penalty = rest[:len(missed)]   # столько же самых слабых
        to_demote = missed + penalty
        for r in to_demote:
            await _demote(db, user_id, r)
        await db.commit()
        return {"passed": False, "demoted": len(to_demote)}
    finally:
        await _release(db)


# ---------------- аудит-экзамен забывания (§2.4-B) ----------------

def _audit_rows(rows):
    """Слова, которым пора на аудит: сертифицированные с audit_due <= now.
    Сортировка по audit_due возрастанию — дольше всех ждавшие первыми (ротация)."""
    now = _now()
    out = [r for r in rows if _is_certified(r) and r.get("audit_due") and r["audit_due"] <= now]
    out.sort(key=lambda r: r.get("audit_due") or "")
    return out


async def build_audit(user_id, cap=AUDIT_CAP, lang="ru"):
    """Собрать аудит-выборку: до cap самых просроченных (audit_due возр.) сертифицированных слов.
    Формат вопросов как в зачётном экзамене (build_gate_exam): {no, pool_id, options:[4]}."""
    import random
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
    finally:
        await _release(db)
    due = _audit_rows(rows)[:cap]
    tr_pool, no_pool, func_pool = _exam_pools(rows, lang)
    questions = []
    for r in due:
        q = _exam_question(r, lang, tr_pool, no_pool, func_pool)
        if q:
            questions.append(q)
    return {"questions": questions, "cap": cap}


async def grade_audit(user_id, answers, lang="ru"):
    """Оценить аудит. answers: [{pool_id, answer}].
    Пословно: верно → следующий аудит ДАЛЬШЕ — интервал растёт между успешными циклами
    (audit_interval × _AUDIT_GROWTH), audit_due = now + новый_интервал; новый интервал помним
    в audit_interval, поэтому 30 → 60 → 120 … при аудите вовремя;
    неверно → де-сертификация (certified=0, status→review, сброс клеток рампы) и слово возвращается
    в очередь изучения (как _demote). Если доля забытых > THROTTLE → throttle=True и притормаживаем
    приток новых на THROTTLE_DAYS (персистится в users.audit_throttle_until, энфорсится new_words_blocked).
    Возвращает {checked, refreshed, forgot, throttle}."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        by_pid = {r["pool_id"]: r for r in rows if _is_certified(r)}
        known = await _known_vocab(db)
        checked = refreshed = forgot = 0
        seen = set()   # дедуп по pool_id: дубль не накручивает checked/refreshed и не разбавляет долю forgot
        for a in (answers or []):
            pid = a.get("pool_id")
            r = by_pid.get(pid)
            if not r or pid in seen:
                continue
            seen.add(pid)
            checked += 1
            ok = _exam_answer_ok(r, lang, a.get("answer"), known)
            if ok:
                # срок растёт между успешными аудитами: новый интервал = прежний × рост.
                # Прежний интервал помним в audit_interval (на сертификации = FIRST_AUDIT_DAYS),
                # поэтому при аудите вовремя получаем 30 → 60 → 120 … (а не залипание на 60).
                prev_interval = r.get("audit_interval") or 0
                if prev_interval <= 0:
                    prev_interval = FIRST_AUDIT_DAYS
                next_days = round(prev_interval * _AUDIT_GROWTH)
                await db.execute(
                    "UPDATE user_words SET audit_due = ?, audit_interval = ? WHERE user_id = ? AND pool_id = ?",
                    (_due_str(next_days), float(next_days), user_id, r["pool_id"]))
                refreshed += 1
            else:
                # забыл → де-сертификация + назад в очередь изучения, снимаем с аудита
                await _demote(db, user_id, r)
                await db.execute(
                    "UPDATE user_words SET audit_due = NULL, audit_interval = 0 WHERE user_id = ? AND pool_id = ?",
                    (user_id, r["pool_id"]))
                forgot += 1
        throttle = bool(checked) and (forgot / checked) > THROTTLE
        if throttle:
            # мягкий тормоз: притормаживаем приток новых на THROTTLE_DAYS (энфорсится на бэкенде)
            await db.execute(
                "UPDATE users SET audit_throttle_until = ? WHERE id = ?",
                (_due_str(THROTTLE_DAYS), user_id))
        await db.commit()
        return {"checked": checked, "refreshed": refreshed, "forgot": forgot, "throttle": throttle}
    finally:
        await _release(db)


