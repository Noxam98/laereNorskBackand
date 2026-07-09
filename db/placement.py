"""Входной тест (placement) + калибровка стартового уровня + досев стартовых слов.
Калибровка консервативна: тест НЕ должен завышать уровень. Вынесено из learning.py
(движок-ядро остаётся там); реэкспортируется через learning.py и db.
"""
from .core import _conn, _release, _now
from .learning import LEVELS, PLACEMENT_INPUT_LEVELS, _fold_loose

# Калибровка входного теста — консервативная (тест НЕ должен завышать уровень):
PLACEMENT_PASS = 0.8     # (легаси, реэкспорт) — сдача считается по PLACEMENT_MAX_MISS, см. grade_placement
PLACEMENT_MIN = 4         # минимум отвеченных вопросов на уровне, иначе уровень не «сдан» (надёжность)
PLACEMENT_MAX_MISS = 5    # допускаем 1 ошибку на каждые N вопросов уровня. Фронт шлёт per=4 → 1 осечка
                          # (3/4) не должна ронять уровень; прежний жёсткий 0.8 требовал идеальный 4/4.
STARTER_GOAL = 20        # сколько слов гарантируем новичку после калибровки, чтобы было что учить


async def get_start_level(user_id):
    db = await _conn()
    try:
        async with db.execute("SELECT start_level FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
        return row["start_level"] if row else None
    finally:
        await _release(db)


async def set_start_level(user_id, level):
    if level not in LEVELS:
        return
    db = await _conn()
    try:
        await db.execute("UPDATE users SET start_level = ? WHERE id = ?", (level, user_id))
        await db.commit()
    finally:
        await _release(db)


async def build_placement(lang="ru", per=8):
    """Набор вопросов входного теста: по `per` слов на каждый уровень A1..C2.
    ДВА типа вопроса (единый контракт с фронтом):
      • choice (A1,A2,B1 — узнавание): {no:<норв. слово, показывается>, level, type:"choice",
        options:[4 перевода на lang]}. Дистракторы — ПРАВДОПОДОБНЫЕ: 3 неверных перевода из слов
        ТОГО ЖЕ уровня и (по возможности) той же части речи (fallback: тот же уровень → любая
        часть речи). Не случайные — чтобы исключение «по очевидности» не завышало балл.
      • input (B2,C1,C2 — продукция): {no:<норв. лемма — КЛЮЧ для грейда>, prompt:<перевод на lang,
        показывается>, level, type:"input"} (без options). Угадать ввод нельзя → честнее на верхах."""
    import random
    from .pool import get_pool_duel_words, get_pool_id
    questions = []
    for lv in LEVELS:
        is_input = lv in PLACEMENT_INPUT_LEVELS
        # широкая выборка слов уровня — и для вопросов, и для дистракторов того же уровня
        words = await get_pool_duel_words(max(per * 6, 40), lv, None)
        random.shuffle(words)
        # переводы уровня по части речи (+ общий список уровня) — пул правдоподобных дистракторов
        by_pos, level_all = {}, []
        for o in words:
            t = (o["translate"].get(lang) or [None])[0]
            if not t:
                continue
            level_all.append((o["norwegian"], t))
            by_pos.setdefault((o.get("part_of_speech") or "").lower(), []).append((o["norwegian"], t))
        picked = 0
        for w in words:
            if picked >= per:
                break
            corr = (w["translate"].get(lang) or [None])[0]
            if not corr:
                continue
            # ТОЧНАЯ запись пула этого смысла (омонимы): несём её id в вопрос, чтобы grade_placement
            # сверял ответ именно с ней, а не переугадывал get_pool_id(no) без pos (др. смысл → «мимо»).
            # Если pos-колонка не проставлена (легаси), get_pool_id(no,pos) вернёт None → падаем на (no).
            qpid = (await get_pool_id(w["norwegian"], w.get("part_of_speech"))
                    or await get_pool_id(w["norwegian"]))
            if is_input:
                # продукция: показываем перевод (prompt), ключ грейда — норвежская лемма (no)
                questions.append({"no": w["norwegian"], "prompt": corr, "level": lv, "type": "input",
                                  "pool_id": qpid})
                picked += 1
                continue
            pos = (w.get("part_of_speech") or "").lower()
            # сначала кандидаты той же части речи (того же уровня), затем добор любым словом уровня
            cand = list(by_pos.get(pos, [])) + level_all
            random.shuffle(cand)
            distract = []
            for o_no, t in cand:
                if o_no == w["norwegian"]:
                    continue
                if t and t != corr and t not in distract:
                    distract.append(t)
                if len(distract) == 3:
                    break
            if len(distract) < 3:
                continue
            opts = distract + [corr]
            random.shuffle(opts)
            questions.append({"no": w["norwegian"], "level": lv, "type": "choice", "options": opts,
                              "pool_id": qpid})
            picked += 1
    random.shuffle(questions)
    return {"questions": questions}


async def grade_placement(user_id, lang, answers):
    """answers: [{no, level, answer, type}]. Оцениваем долю верных по уровням и КОНСЕРВАТИВНО
    оцениваем стартовый уровень: идём снизу вверх и засчитываем уровень только если на нём
    отвечено достаточно (>= PLACEMENT_MIN) и ошибок не больше 1 на каждые PLACEMENT_MAX_MISS вопросов
    (одна осечка при per=4 не роняет уровень); останавливаемся на ПЕРВОМ несданном. Проверка — по пулу.
    Сверка по типу вопроса:
      • type=="choice": верно если answer ∈ переводам слова no на язык lang (узнавание);
      • type=="input": верно если введённое совпало с НОРВЕЖСКОЙ леммой no (или любым из
        translate.no) СНИСХОДИТЕЛЬНО — через _fold_loose (lower/trim, å→a ø→o æ→ae, срез диакритик).
    Тип берём из ответа; если его нет — выводим из уровня (верхние → input), чтобы не падать на
    старых клиентах."""
    from .pool import get_pool_id, get_pool_by_id
    per_lvl = {lv: {"ok": 0, "total": 0} for lv in LEVELS}
    for a in (answers or []):
        if not isinstance(a, dict):   # мусор в answers ([str]/[null]) не должен ронять грейд (500)
            continue
        lv = a.get("level")
        if lv not in per_lvl:
            continue
        per_lvl[lv]["total"] += 1
        no = str(a.get("no") or "")
        # омоним: грейдим по ТОЧНОЙ записи пула, которую построил build_placement (carried pool_id),
        # иначе get_pool_id(no) без pos мог взять ДРУГОЙ смысл (др. переводы) → верный ответ «мимо».
        cid = a.get("pool_id")
        try:
            pid = int(cid) if cid is not None else None
        except (TypeError, ValueError):
            pid = None
        if not pid:                    # старый клиент без pool_id → прежний путь (по слову, без pos)
            pid = await get_pool_id(no)
        if not pid:
            continue
        p = await get_pool_by_id(pid)
        translate = ((p or {}).get("data") or {}).get("translate", {})
        qtype = a.get("type") or ("input" if lv in PLACEMENT_INPUT_LEVELS else "choice")
        if qtype == "input":
            # продукция: введённое сравниваем с норвежской леммой no и всеми вариантами translate.no
            ans = _fold_loose(str(a.get("answer") or ""))
            keys = {_fold_loose(no)} | {_fold_loose(t) for t in (translate.get("no") or [])}
            keys.discard("")
            if ans and ans in keys:
                per_lvl[lv]["ok"] += 1
        else:
            tr = translate.get(lang) or []
            ans = str(a.get("answer") or "").strip().lower()
            if ans and any(ans == (t or "").strip().lower() for t in tr):
                per_lvl[lv]["ok"] += 1
    # уровень = высший пройденный подряд снизу; стоп на первом несданном (консервативно, вниз)
    level = "A1"
    for lv in LEVELS:
        d = per_lvl[lv]
        passed = d["total"] >= PLACEMENT_MIN and (d["total"] - d["ok"]) <= max(1, d["total"] // PLACEMENT_MAX_MISS)
        if passed:
            level = lv
        else:
            break
    await set_start_level(user_id, level)
    # Плейсмент задаёт BASELINE «знаю»: сдал уровень L → топ-частотный КОР тиров НИЖЕ L считаем известным
    # (сдача уровня подразумевает владение нижними — CEFR-лестница). known (НЕ mastered): уровень не двигает,
    # обратимо, из подбора уходит (suggest_words.have чтит known). Иначе placed-B1 переучивал ~750 базовых.
    await _seed_known_baseline(user_id, level)
    return {"level": level, "perLevel": per_lvl}


async def _seed_known_baseline(user_id, level):
    """Пометить known топ-частотные слова тиров СТРОГО НИЖЕ сданного (кап = LEVEL_TARGETS тира по частоте;
    редкий хвост тира остаётся учащимся). INSERT OR IGNORE по PK (user_id,pool_id) НЕ трогает уже
    выученные/учащиеся/known — прогресс и активное изучение сохранены; сеются только нетронутые слова."""
    from .learning import LEVEL_TARGETS   # ленивый импорт (избегаем цикла на загрузке модуля)
    li = LEVELS.index(level) if level in LEVELS else 0
    if li == 0:
        return
    db = await _conn()
    try:
        for t in LEVELS[:li]:
            cap = LEVEL_TARGETS.get(t, 300)
            await db.execute(
                "INSERT OR IGNORE INTO user_words (user_id, pool_id, created_at, known) "
                "SELECT ?, id, ?, 1 FROM word_pool "
                "WHERE level = ? AND data IS NOT NULL AND COALESCE(learn_excluded,0)=0 AND COALESCE(approved,1)=1 "
                "ORDER BY freq IS NULL, freq DESC LIMIT ?", (user_id, _now(), t, cap))
        await db.commit()
    finally:
        await _release(db)


async def seed_starter(user_id, level, target=STARTER_GOAL):
    """Досыпать новых слов под уровень до target, если у пользователя их почти нет.
    Вызывается после входного теста / самооценки — чтобы Учёба не была пустой."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(DISTINCT pool_id) AS n FROM dict_words WHERE dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)",
            (user_id,)) as cur:
            row = await cur.fetchone()
            have = (row["n"] if row else 0) or 0
    finally:
        await _release(db)
    need = target - have
    if need <= 0:
        return {"seeded": 0, "had": have}
    from .learning import suggest_words   # ленивый: suggest_words вынесен в learning_suggest (грузится позже placement)
    # allow_func=False: стартовый набор — только контентные слова; служебные вводит build_session со
    # своим пословным гейтом (иначе в «20 стартовых» попадали og/i/ikke, которые A1-гейт не пускает).
    res = await suggest_words(user_id, count=need, level=level, allow_func=False)
    return {"seeded": res.get("added", 0), "had": have}
