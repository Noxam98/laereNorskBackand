"""Зачётный экзамен-ворота (§2.4-A) + аудит-экзамен забывания (§2.4-B).
Пейсинг притока новых слов и проверка забывания сертифицированных. Вынесено из learning.py.
ВАЖНО: _is_certified и _audit_throttled нужны ядру (build_session/get_due/learning_stats) —
они определены здесь и реэкспортируются обратно в learning (см. конец learning.py).
"""
import json
import math
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


def _load_data(r):
    """Безопасный разбор JSON-колонки data строки пула: одна битая строка не должна ронять
    сборку/грейд всего экзамена (try/except → {}, как в остальных местах кодовой базы)."""
    try:
        return json.loads(r["data"]) if r.get("data") else {}
    except Exception:
        return {}


def _exam_pools(rows, lang):
    """Пулы дистракторов: переводы (no2int), норвежские слова (int2no), норвежские служебные (cloze)."""
    tr_pool, no_pool, func_pool = [], [], []
    for r in rows:
        data = _load_data(r)
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
    data = _load_data(r)
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
        # мало норв. дистракторов → откатываемся на другой формат (см. ниже)
    # no2int: норв. слово (видимый вопрос) → выбрать перевод
    distract = _pick_distract(tr_pool, {corr_tr}, 3)
    if len(distract) == 3:
        opts = distract + [corr_tr]; random.shuffle(opts)
        return {"type": "no2int", "no": no, "pool_id": r["pool_id"], "options": opts}
    # последний резерв — ввод (нужна только подсказка-перевод, дистракторы не требуются). Так слово
    # с переводом ВСЕГДА даёт вопрос → выборка не «худеет» из-за нехватки дистракторов, и порог
    # сдачи (см. _pass_threshold) детерминирован и совпадает у build и grade (_question_buildable).
    return {"type": "input", "prompt": corr_tr, "pool_id": r["pool_id"]}


def _question_buildable(r, lang, func_pool):
    """Детерминированно (без random-roll): даст ли слово вопрос в _exam_question.
    Годится, если есть перевод (тогда input-резерв возможен всегда) ИЛИ слово cloze-пригодно
    (служебное с примером и 3 служебными дистракторами). Нужно, чтобы grade знал реальный размер
    выборки и не требовал недостижимый фикс-PASS, когда вопросов сгенерилось меньше SAMPLE."""
    data = _load_data(r)
    tr = data.get("translate", {}).get(lang) or []
    if tr and tr[0]:
        return True
    no = r["norwegian"]
    ex = data.get("example") or {}
    ex_no = (ex.get("no") or "").strip() if isinstance(ex, dict) else ""
    if is_function_word(no, data) and ex_no:
        if _blank_example(ex_no, no) and len(_pick_distract(func_pool, {no}, 3)) == 3:
            return True
    return False


def _pass_threshold(n):
    """Эффективный порог сдачи ворот для выборки из n вопросов: держим долю ~90% (как штатное
    PASS/SAMPLE = 27/30), но не выше PASS. Если из пачки строится < SAMPLE вопросов, фикс-PASS
    делал ворота непроходимыми (приток новых заперт навсегда) — здесь порог масштабируется.
    n<=0 → 1 (недостижимо: correct_n=0 < 1, пачку не сертифицируем на пустом экзамене)."""
    if n <= 0:
        return 1
    return min(PASS, max(1, math.ceil(0.9 * n)))


_KNOWN_VOCAB = None


async def _known_vocab(db):
    """Множество нормализованных норв. слов пула — словарь-страж для нечёткой сверки (ввод, совпавший
    с ДРУГИМ известным словом, не считаем опечаткой). Кэш на процесс (новые слова редки)."""
    global _KNOWN_VOCAB
    if _KNOWN_VOCAB is None:
        async with db.execute("SELECT norwegian FROM word_pool") as cur:
            _KNOWN_VOCAB = {fuzzy.normalize(r["norwegian"]) for r in await cur.fetchall() if r["norwegian"]}
    return _KNOWN_VOCAB


def _exam_answer_ok(r, lang, ans, known=None, accept_translation=True, accept_word_forms=True):
    """Нечёткая сверка (fuzzy.py): ответ достаточно близок к «своей» форме слова. Со словарём-стражем
    (known) ввод, равный ДРУГОМУ известному слову, отклоняется (это не опечатка).

    accept_translation=False (тип input) — принимаем ТОЛЬКО норвежское слово и его словоформы, НЕ
    показанный перевод: иначе юзер печатает показанную ему подсказку-перевод и получает «верно».
    accept_word_forms=False (тип no2int) — норвежская лемма ВИДИМА в самом вопросе (это промпт),
    поэтому норвежские словоформы НЕ принимаем: иначе клиент эхом промпта всегда «сдаёт» (обход
    ворот забывания). Принимаем только перевод.
    Оба True (типы-выбор int2no/cloze) — варианты без подписей, ответом может быть перевод ИЛИ
    норвежское слово, поэтому сверка остаётся типонезависимой."""
    data = _load_data(r)
    try:
        fcol = json.loads(r["forms"]) if r.get("forms") else None
    except Exception:
        fcol = None
    acc = list(data.get("translate", {}).get(lang) or []) if accept_translation else []
    if accept_word_forms:
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
    # порог сдачи масштабируем под фактический размер выборки (см. _pass_threshold): если из пачки
    # построилось < SAMPLE вопросов, фикс-PASS сделал бы ворота непроходимыми.
    return {"questions": questions, "sample": SAMPLE, "pass": _pass_threshold(len(questions))}


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
    """Оценить зачётный экзамен. answers: [{pool_id, answer, type?}].
    Верных ≥ эффективного порога (_pass_threshold от реального размера выборки, ≤ PASS) →
    сертифицировать всю текущую несданную пачку (certified=1), {passed:True}.
    Иначе провал: демоут каждого промаха (mastered→review) + столько же самых слабых слов
    пачки по strength (штраф ×2 промаха). {passed:False, demoted:N}.
    type (опц.): для type=='input' показанный перевод НЕ засчитывается (только норв. формы)."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        pack = _pack_rows(rows)
        by_pid = {r["pool_id"]: r for r in pack}
        known = await _known_vocab(db)
        # эффективный порог = ~90% от реально построимой выборки (капнутой SAMPLE), но не выше PASS.
        # _question_buildable детерминирует размер выборки так же, как её строит build_gate_exam,
        # поэтому порог совпадает у build и grade и всегда достижим (иначе фикс-PASS запирал ворота).
        _, _, func_pool = _exam_pools(rows, lang)
        buildable = sum(1 for r in pack if _question_buildable(r, lang, func_pool))
        need = _pass_threshold(min(SAMPLE, buildable))
        # сверка ответов — нечётко, по «своей» форме слова (для input перевод-подсказку не принимаем)
        correct_n = 0
        missed_pids = []
        seen = set()   # дедуп по pool_id: одна карточка засчитывается ОДИН раз. Иначе повтор верного
        for a in (answers or []):   # ответа ×PASS подделывал бы сдачу ворот (сертификацию всей пачки 50–100 слов)
            if not isinstance(a, dict):   # мусор в answers ([str], [null]) не должен ронять грейд (500)
                continue
            pid = a.get("pool_id")
            r = by_pid.get(pid)
            if not r or pid in seen:
                continue
            seen.add(pid)
            qtype = a.get("type")
            accept_tr = qtype != "input"     # input: показанный перевод НЕ засчитываем
            accept_wf = qtype != "no2int"    # no2int: норв. лемма — видимый вопрос → эхо промпта не сдаём
            ans_txt = str(a.get("answer") or "").strip()   # безопасная коэрция (не .strip() по не-str)
            if _exam_answer_ok(r, lang, ans_txt, known,
                               accept_translation=accept_tr, accept_word_forms=accept_wf):
                correct_n += 1
            else:
                missed_pids.append(pid)

        # мутацию user_words держим под пер-юзер локом (Фикс #7): read-modify-write иначе теряет
        # апдейт против конкурентного /answer того же юзера (второй commit затирает первый).
        from .learning import _user_lock   # ленивый: цикл learning ↔ exams
        async with _user_lock(user_id):
            # buildable==0 → выборка пуста: не сертифицируем даже при correct_n>=need (need=1 на пустом
            # экзамене был бы достижим эхом норв. слов). _pass_threshold(0) оставляем =1 (контракт).
            if buildable > 0 and correct_n >= need:
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
        # грейдим ТОЛЬКО реально выданный набор (те же самые просроченные до AUDIT_CAP, что вернул
        # build_audit) — иначе клиент мог бы прислать pool_id любого сертифицированного слова и
        # двигать его audit_due / де-сертифицировать не-due слово (клиентские id не сверялись с выданными).
        by_pid = {r["pool_id"]: r for r in _audit_rows(rows)[:AUDIT_CAP]}
        known = await _known_vocab(db)
        checked = refreshed = forgot = 0
        seen = set()   # дедуп по pool_id: дубль не накручивает checked/refreshed и не разбавляет долю forgot
        # мутацию user_words держим под пер-юзер локом (Фикс #7): иначе теряется апдейт против
        # конкурентного /answer того же юзера (второй commit затирает первый).
        from .learning import _user_lock   # ленивый: цикл learning ↔ exams
        async with _user_lock(user_id):
            for a in (answers or []):
                if not isinstance(a, dict):   # мусор в answers не должен ронять грейд (500)
                    continue
                pid = a.get("pool_id")
                r = by_pid.get(pid)
                if not r or pid in seen:
                    continue
                seen.add(pid)
                checked += 1
                qtype = a.get("type")
                accept_tr = qtype != "input"     # input: показанный перевод НЕ засчитываем
                accept_wf = qtype != "no2int"    # no2int: норв. лемма — видимый вопрос → эхо не сдаём
                ans_txt = str(a.get("answer") or "").strip()   # безопасная коэрция (не .strip() по не-str)
                ok = _exam_answer_ok(r, lang, ans_txt, known,
                                     accept_translation=accept_tr, accept_word_forms=accept_wf)
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


