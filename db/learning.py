"""«Учёба» — слой интервальных повторений (SRS) над всеми словами пользователя.
Слова берутся из словарей (dict_words → word_pool), состояние — в user_words.
Статусы вычисляются на чтении из силы/попыток; archived — ручной флаг («я это знаю»)."""
import json
from datetime import datetime, timedelta
from .core import _conn, _release, _now

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
# Сколько слов уровня надо «выучить» (mastered/архив), чтобы закрыть уровень и идти выше.
# Жёсткая шкала, близкая к реальным объёмам словаря CEFR — каждый уровень заметно тяжелее.
LEVEL_TARGETS = {"A1": 250, "A2": 500, "B1": 1000, "B2": 2000, "C1": 3500, "C2": 5000}
_INTERVAL_CAP = 365
_FAST_SEC = 3.0   # быстрый верный ответ → слово «лёгкое», сильнее растёт
# «Выучено» = в КАЖДОМ виде игры пройдено N раз подряд без ошибок.
# study — пассивный (карточки): «пройти» = просмотреть N раз (ошибок там нет).
REQUIRED_MODES = ["study", "choice", "input"]
MASTER_PER_MODE = 3                     # сколько раз подряд без ошибок нужно в каждом виде
# Прогресс — скользящее окно: новые попытки вытесняют старые (ёмкость). Сила слова считается
# по последним CAPACITY попыткам, а не за всю историю — старые успехи «выпадают» со временем.
CAPACITY = 8
DAILY_GOAL = 20   # дневная цель по умолчанию (слов/ответов)
_LEVEL_ORDER = {lv: i for i, lv in enumerate(LEVELS)}


async def _activity_metrics(db, user_id):
    """Стрик, дневная цель, точность по журналу активности."""
    from datetime import date, timedelta
    async with db.execute("SELECT day, answers, correct FROM user_activity WHERE user_id = ?", (user_id,)) as cur:
        rows = {r["day"]: (r["answers"], r["correct"]) for r in await cur.fetchall()}
    today = date.today()
    today_s = today.isoformat()
    done = rows.get(today_s, (0, 0))[0]
    # стрик: цепочка активных дней, заканчивающаяся сегодня или вчера (грейс на «ещё не занимался сегодня»)
    active = set(rows.keys())
    start = today if today_s in active else (today - timedelta(days=1))
    streak = 0
    d = start
    while d.isoformat() in active:
        streak += 1
        d -= timedelta(days=1)
    # точность за 30 дней
    cut = (today - timedelta(days=30)).isoformat()
    a30 = sum(v[0] for k, v in rows.items() if k >= cut)
    c30 = sum(v[1] for k, v in rows.items() if k >= cut)
    accuracy = round(100 * c30 / a30) if a30 else None
    return {"streak": streak, "today": {"done": done, "goal": DAILY_GOAL}, "accuracy": accuracy}


def _push(window, bit, cap):
    """Добавить исход (1/0) в строку-окно и обрезать слева до ёмкости cap."""
    return (window + bit)[-cap:]


def _strength_from(hist):
    return round(100 * hist.count("1") / len(hist)) if hist else 0


def _today():
    return datetime.utcnow().date()


def _due_str(days):
    return (datetime.utcnow() + timedelta(days=days)).isoformat()


def _mastered_by_modes(modes):
    """Выучено: в каждом тестовом режиме последние MASTER_PER_MODE попыток — все верные (без ошибок)."""
    return all((modes or {}).get(m, "") == "1" * MASTER_PER_MODE for m in REQUIRED_MODES)


def status_of(row, modes=None):
    """Статус слова из его состояния."""
    if row.get("archived"):
        return "archived"
    attempts = (row.get("correct") or 0) + (row.get("incorrect") or 0)
    strength = row.get("strength") or 0
    if attempts == 0:
        return "new"
    if _mastered_by_modes(modes):
        return "mastered"
    if strength < 40 and (row.get("incorrect") or 0) >= 2:
        return "weak"
    if (row.get("reps") or 0) >= 1:
        return "review"
    return "learning"


def _is_due(row):
    due = row.get("due_at")
    return bool(due) and due <= _now() and not row.get("archived")


# ---------------- запись результата ответа (SRS) ----------------

async def apply_result(user_id: int, pool_id: int, correct: bool, elapsed: float = None, mode: str = None):
    """Обновить состояние слова после ответа (создаёт строку при первом ответе).
    mode — режим (choice/input/study/…): копим серию верных ПОДРЯД в каждом режиме;
    ошибка обнуляет серию режима. «Выучено» = серия ≥ MASTER_PER_MODE во всех тестовых режимах."""
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM user_words WHERE user_id = ? AND pool_id = ?", (user_id, pool_id)) as cur:
            r = await cur.fetchone()
        st = dict(r) if r else {"strength": 0, "reps": 0, "lapses": 0, "ease": 2.5, "interval_days": 0,
                                "correct": 0, "incorrect": 0, "streak": 0, "archived": 0, "modes": None}
        modes = {}
        try:
            modes = json.loads(st.get("modes") or "{}")
        except Exception:
            modes = {}
        # окна попыток — нормализуем (старый формат с числами игнорим)
        modes = {k: v for k, v in modes.items() if isinstance(v, str)}
        bit = "1" if correct else "0"
        ease = st["ease"]; interval = st["interval_days"]
        modes["hist"] = _push(modes.get("hist", ""), bit, CAPACITY)   # общее окно для силы
        if mode:
            modes[mode] = _push(modes.get(mode, ""), bit, MASTER_PER_MODE)  # окно режима для «без ошибок»
        strength = _strength_from(modes["hist"])
        if correct:
            fast = elapsed is not None and elapsed <= _FAST_SEC
            ease = min(3.0, ease + (0.08 if fast else 0.04))
            interval = 1 if interval < 1 else min(_INTERVAL_CAP, round(interval * ease))
            if fast and interval < 1:
                interval = 2
            st["reps"] += 1; st["correct"] += 1; st["streak"] += 1
            due = _due_str(interval)
        else:
            ease = max(1.3, ease - 0.2)
            interval = 1
            st["lapses"] += 1; st["incorrect"] += 1; st["streak"] = 0
            due = _due_str(1)
        modes_json = json.dumps(modes, ensure_ascii=False)
        await db.execute("""
            INSERT INTO user_words (user_id, pool_id, strength, reps, lapses, ease, interval_days, due_at,
                                    correct, incorrect, streak, archived, modes, last_seen, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, pool_id) DO UPDATE SET
                strength=excluded.strength, reps=excluded.reps, lapses=excluded.lapses, ease=excluded.ease,
                interval_days=excluded.interval_days, due_at=excluded.due_at, correct=excluded.correct,
                incorrect=excluded.incorrect, streak=excluded.streak, modes=excluded.modes, last_seen=excluded.last_seen
        """, (user_id, pool_id, strength, st["reps"], st["lapses"], ease, interval, due,
              st["correct"], st["incorrect"], st["streak"], st.get("archived", 0), modes_json, _now(), _now()))
        # дневная активность (для стрика/цели/точности/хитмапа)
        day = _now()[:10]
        await db.execute("""
            INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,1,?)
            ON CONFLICT(user_id, day) DO UPDATE SET answers = answers + 1, correct = correct + ?
        """, (user_id, day, 1 if correct else 0, 1 if correct else 0))
        await db.commit()
        return {"ok": True, "strength": strength, "due_at": due, "modes": modes, "mastered": _mastered_by_modes(modes)}
    finally:
        await _release(db)


async def set_status(user_id: int, pool_id: int, action: str):
    """action: know (в архив, сила 100) | reset (сброс) | unarchive (вернуть в ротацию)."""
    db = await _conn()
    try:
        if action == "know":
            m = {mm: "1" * MASTER_PER_MODE for mm in REQUIRED_MODES}
            m["hist"] = "1" * CAPACITY
            fields = "archived=1, strength=100, reps=MAX(reps,3), modes=?, due_at=?"
            args = (json.dumps(m, ensure_ascii=False), _due_str(120))
        elif action == "reset":
            fields = "archived=0, strength=0, reps=0, lapses=0, ease=2.5, interval_days=0, correct=0, incorrect=0, streak=0, modes=NULL, due_at=NULL"
            args = ()
        elif action == "unarchive":
            fields = "archived=0, due_at=?"
            args = (_due_str(1),)
        else:
            return {"error": "bad action"}
        # гарантируем строку
        await db.execute("INSERT OR IGNORE INTO user_words (user_id, pool_id, created_at) VALUES (?,?,?)", (user_id, pool_id, _now()))
        await db.execute(f"UPDATE user_words SET {fields} WHERE user_id = ? AND pool_id = ?", (*args, user_id, pool_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


# ---------------- выборка слов пользователя ----------------

async def _fetch_user_words(db, user_id):
    """Все слова пользователя (уникальные по пулу) + состояние SRS + темы."""
    async with db.execute("""
        SELECT wp.id AS pool_id, wp.norwegian, wp.data, wp.level, (wp.tts IS NOT NULL) AS has_tts,
               uw.strength, uw.reps, uw.lapses, uw.ease, uw.interval_days, uw.due_at,
               uw.correct, uw.incorrect, uw.streak, uw.archived, uw.modes, uw.last_seen
        FROM (SELECT DISTINCT pool_id FROM dict_words
              WHERE dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)) up
        JOIN word_pool wp ON wp.id = up.pool_id
        LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
    """, (user_id, user_id)) as cur:
        rows = [dict(r) for r in await cur.fetchall()]
    # темы пачкой
    if rows:
        ids = [r["pool_id"] for r in rows]
        marks = ",".join("?" for _ in ids)
        topics = {}
        async with db.execute(f"SELECT pool_id, topic FROM word_topics WHERE pool_id IN ({marks})", ids) as cur:
            for tr in await cur.fetchall():
                topics.setdefault(tr["pool_id"], []).append(tr["topic"])
        for r in rows:
            r["topics"] = topics.get(r["pool_id"], [])
    return rows


def _shape(r):
    data = json.loads(r["data"]) if r.get("data") else {}
    try:
        modes = json.loads(r.get("modes") or "{}")
    except Exception:
        modes = {}
    return {
        "pool_id": r["pool_id"], "no": r["norwegian"],
        "translate": data.get("translate", {}), "part_of_speech": data.get("part_of_speech", ""),
        "level": r.get("level"), "topics": r.get("topics", []), "hasTts": bool(r.get("has_tts")),
        "status": status_of(r, modes), "strength": r.get("strength") or 0, "due_at": r.get("due_at"),
        "modes": modes, "correct": r.get("correct") or 0, "incorrect": r.get("incorrect") or 0, "due": _is_due(r),
        "last_seen": r.get("last_seen"),
    }


async def get_learning(user_id, status=None, level=None, topic=None, q=None, sort="strength", limit=200, offset=0):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
    finally:
        await _release(db)
    items = [_shape(r) for r in rows]
    if status and status != "all":
        items = [w for w in items if w["status"] == status]
    if level:
        items = [w for w in items if (w["level"] or "") == level]
    if topic:
        items = [w for w in items if topic in (w["topics"] or [])]
    if q:
        ql = q.strip().lower()
        def hit(w):
            if ql in (w["no"] or "").lower():
                return True
            for arr in (w["translate"] or {}).values():
                if any(ql in (s or "").lower() for s in arr):
                    return True
            return False
        items = [w for w in items if hit(w)]
    if sort == "alpha":
        items.sort(key=lambda w: (w["no"] or "").lower())
    elif sort == "due":
        items.sort(key=lambda w: (w["due_at"] or "9999"))
    else:  # strength (слабые сверху)
        items.sort(key=lambda w: w["strength"])
    total = len(items)
    return {"total": total, "words": items[offset:offset + limit]}


async def get_due(user_id, limit=20):
    """Очередь Smart Review: просроченные (по due) + слабые + немного новых."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
    finally:
        await _release(db)
    items = [_shape(r) for r in rows]
    due = sorted([w for w in items if w["due"]], key=lambda w: w["due_at"] or "")
    weak = sorted([w for w in items if w["status"] == "weak" and not w["due"]], key=lambda w: w["strength"])
    new = [w for w in items if w["status"] == "new"]
    queue, seen = [], set()
    for w in due + weak + new:
        if w["pool_id"] in seen:
            continue
        seen.add(w["pool_id"]); queue.append(w)
        if len(queue) >= limit:
            break
    return {"words": queue}


async def learning_stats(user_id):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        activity = await _activity_metrics(db, user_id)
    finally:
        await _release(db)
    items = [_shape(r) for r in rows]
    by_status = {}
    by_level = {lv: {"total": 0, "mastered": 0, "target": LEVEL_TARGETS[lv], "done": False} for lv in LEVELS}
    due_count = 0
    for w in items:
        by_status[w["status"]] = by_status.get(w["status"], 0) + 1
        if w["due"]:
            due_count += 1
        lv = w["level"] if w["level"] in by_level else None
        if lv:
            by_level[lv]["total"] += 1
            if w["status"] in ("mastered", "archived"):
                by_level[lv]["mastered"] += 1
    for lv in LEVELS:
        by_level[lv]["done"] = by_level[lv]["mastered"] >= by_level[lv]["target"]
    # текущий уровень = первый незакрытый по цели, но не ниже уровня из входного теста
    computed = next((lv for lv in LEVELS if not by_level[lv]["done"]), "C2")
    start = await get_start_level(user_id)
    current = computed
    if start and _LEVEL_ORDER.get(start, 0) > _LEVEL_ORDER.get(computed, 0):
        current = start
    cur = by_level[current]
    to_next = max(0, cur["target"] - cur["mastered"])
    # ретеншн = доля выученных среди (выучено+слабых); «выучено за неделю» — по last_seen
    from datetime import datetime, timedelta
    mastered_n = (by_status.get("mastered", 0) + by_status.get("archived", 0))
    weak_n = by_status.get("weak", 0)
    retention = round(100 * mastered_n / (mastered_n + weak_n)) if (mastered_n + weak_n) else None
    week_cut = (datetime.utcnow() - timedelta(days=7)).isoformat()
    mastered_week = sum(1 for w in items if w["status"] in ("mastered", "archived") and (w.get("last_seen") or "") >= week_cut)
    return {"total": len(items), "due": due_count, "byStatus": by_status, "byLevel": by_level,
            "currentLevel": current, "toNextLevel": to_next, "startLevel": start, "placed": bool(start),
            "streak": activity["streak"], "today": activity["today"], "accuracy": activity["accuracy"],
            "retention": retention, "masteredWeek": mastered_week}


# ---------------- входной тест (placement) ----------------

async def get_activity(user_id, days=119):
    """Дневная активность за период (для хитмапа). [{day, answers, correct}]."""
    from datetime import date, timedelta
    cut = (date.today() - timedelta(days=days)).isoformat()
    db = await _conn()
    try:
        async with db.execute("SELECT day, answers, correct FROM user_activity WHERE user_id=? AND day>=? ORDER BY day", (user_id, cut)) as cur:
            return {"days": [dict(r) for r in await cur.fetchall()]}
    finally:
        await _release(db)


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


async def build_placement(lang="ru", per=4):
    """Набор вопросов входного теста: по `per` слов на каждый уровень A1..C2.
    Вопрос: норвежское слово + 4 варианта перевода на язык lang (без отметки верного)."""
    import random
    from .pool import get_pool_duel_words
    questions = []
    # запас переводов для дистракторов
    pool_all = await get_pool_duel_words(400, None, None)
    for lv in LEVELS:
        words = await get_pool_duel_words(per * 3, lv, None)
        picked = 0
        for w in words:
            if picked >= per:
                break
            corr = (w["translate"].get(lang) or [None])[0]
            if not corr:
                continue
            distract = []
            for o in pool_all:
                if o["norwegian"] == w["norwegian"]:
                    continue
                t = (o["translate"].get(lang) or [None])[0]
                if t and t != corr and t not in distract:
                    distract.append(t)
                if len(distract) == 3:
                    break
            if len(distract) < 3:
                continue
            opts = distract + [corr]
            random.shuffle(opts)
            questions.append({"no": w["norwegian"], "level": lv, "options": opts})
            picked += 1
    random.shuffle(questions)
    return {"questions": questions}


async def grade_placement(user_id, lang, answers):
    """answers: [{no, level, answer}]. Оцениваем долю верных по уровням, оцениваем уровень
    (поднимаемся, пока уровень сдан ≥60%), сохраняем как стартовый. Проверка ответов — по пулу."""
    from .pool import get_pool_id, get_pool_by_id
    per_lvl = {lv: {"ok": 0, "total": 0} for lv in LEVELS}
    for a in (answers or []):
        lv = a.get("level")
        if lv not in per_lvl:
            continue
        per_lvl[lv]["total"] += 1
        pid = await get_pool_id(a.get("no") or "")
        if not pid:
            continue
        p = await get_pool_by_id(pid)
        tr = ((p or {}).get("data") or {}).get("translate", {}).get(lang) or []
        ans = (a.get("answer") or "").strip().lower()
        if ans and any(ans == (t or "").strip().lower() for t in tr):
            per_lvl[lv]["ok"] += 1
    # уровень = высший пройденный подряд (≥60%), иначе A1
    level = "A1"
    for lv in LEVELS:
        d = per_lvl[lv]
        if d["total"] and d["ok"] / d["total"] >= 0.6:
            level = lv
        else:
            break
    await set_start_level(user_id, level)
    return {"level": level, "perLevel": per_lvl}


async def estimate_level(user_id):
    """Рабочий уровень для подсказок — текущий уровень по целям (первый незакрытый)."""
    return (await learning_stats(user_id))["currentLevel"]


async def suggest_words(user_id, count=10, level=None):
    """«Докинуть слов»: добавить в словарь пользователя новые слова пула по его уровню,
    которых у него ещё нет. Возвращает добавленные. Импорт здесь, чтобы избежать циклов."""
    from .pool import get_pool_duel_words, get_pool_id
    from .dictionaries import add_word_to_dict
    lvl = level if level in LEVELS else await estimate_level(user_id)
    db = await _conn()
    try:
        async with db.execute("SELECT DISTINCT pool_id FROM dict_words WHERE dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)", (user_id,)) as cur:
            have = {r["pool_id"] for r in await cur.fetchall()}
        # целевой словарь: текущий пользователя или первый
        async with db.execute("SELECT current_dict FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
        cur_name = row["current_dict"] if row else None
        async with db.execute("SELECT id, name FROM dictionaries WHERE user_id = ? ORDER BY created_at, id", (user_id,)) as cur:
            dicts = [dict(r) for r in await cur.fetchall()]
    finally:
        await _release(db)
    if not dicts:
        return {"added": 0, "words": [], "level": lvl, "error": "no_dict"}
    target = next((d for d in dicts if d["name"] == cur_name), dicts[0])
    # кандидаты по уровню, с запасом, исключаем уже имеющиеся
    cand = await get_pool_duel_words(max(count * 4, 40), lvl, None)
    added = []
    for w in cand:
        if len(added) >= count:
            break
        pid = await get_pool_id(w["norwegian"])
        if not pid or pid in have:
            continue
        res = await add_word_to_dict(user_id, target["id"], pid)
        if res.get("id") and not res.get("duplicate"):
            added.append({"no": w["norwegian"], "translate": w.get("translate", {})})
            have.add(pid)
    return {"added": len(added), "words": added, "level": lvl, "dict": target["name"]}
