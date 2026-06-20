"""«Учёба» — слой интервальных повторений (SRS) над всеми словами пользователя.
Слова берутся из словарей (dict_words → word_pool), состояние — в user_words.
Статусы вычисляются на чтении из силы/попыток; archived — ручной флаг («я это знаю»)."""
import json
import unicodedata
from datetime import datetime, timedelta
from .core import _conn, _release, _now

LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
# Верхние уровни входного теста идут на ВВОД (продукция норвежского), а не «выбор из 4»
# (узнавание). Выбор угадывается, ввод — нет, поэтому продукция честнее на высоких уровнях.
PLACEMENT_INPUT_LEVELS = {"B2", "C1", "C2"}


def _fold_loose(s: str) -> str:
    """Снисходительная нормализация для сверки ВВОДА с норвежской леммой (как foldLoose на фронте):
    lower/trim, å→a ø→o æ→ae и срез прочих диакритик. Пустую/None → ''."""
    s = (s or "").strip().lower()
    s = s.replace("å", "a").replace("ø", "o").replace("æ", "ae")
    # срезаем остальные диакритики (é→e, ü→u, …): раскладываем и убираем combining-метки
    s = "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))
    return s
# Сколько слов уровня надо «выучить» (mastered/архив), чтобы закрыть уровень и идти выше.
# Жёсткая шкала, близкая к реальным объёмам словаря CEFR — каждый уровень заметно тяжелее.
LEVEL_TARGETS = {"A1": 250, "A2": 500, "B1": 1000, "B2": 2000, "C1": 3500, "C2": 5000}
_INTERVAL_CAP = 365
_FAST_SEC = 3.0   # быстрый верный ответ → слово «лёгкое», сильнее растёт
# «Выучено» = пройдена вся РАМПА сложности: 4 клетки (тип игры × направление), по 1 верному
# в каждой. Рампа «узнавание → сборка → ввод», всё ведёт к производству норвежского.
# study (карточки) — пассивный шаг 0, в зачёт не идёт.
# Каждая клетка хранит признак прохождения: '1' — пройдена, '' — не пройдена/сброшена ошибкой.
REQUIRED_CELLS = ["choice_no2int", "choice_int2no", "build_int2no", "input_int2no"]
# build (собери из букв) осмыслен только в направлении родной→норв
_DIR_ALLOWED = {"choice": ("no2int", "int2no"), "build": ("int2no",), "input": ("no2int", "int2no")}
# Прогресс — скользящее окно: новые попытки вытесняют старые (ёмкость). Сила слова считается
# по последним CAPACITY попыткам, а не за всю историю — старые успехи «выпадают» со временем.
CAPACITY = 8
DAILY_GOAL = 20   # дневная цель по умолчанию (слов/ответов)
_LEVEL_ORDER = {lv: i for i, lv in enumerate(LEVELS)}

# --- Зачётный экзамен-ворота (§2.4-A): пейсит приток новых слов ---
PACK_FIRST = 50   # первый порог (до первой сертификации)
PACK = 100        # порог последующих пачек
SAMPLE = 30       # сколько случайных вопросов в выборке экзамена
PASS = 27         # сколько нужно верных, чтобы сдать

# --- Аудит-экзамен забывания (§2.4-B): ловит забывание сертифицированных слов ---
FIRST_AUDIT_DAYS = 30   # первый аудит слова — через 30 дней после сертификации
AUDIT_CAP = 20          # потолок аудит-сессии (берём не более стольких самых просроченных)
THROTTLE = 0.4          # доля забытых выше которой — мягкий тормоз притока новых
_AUDIT_GROWTH = 2.0     # во сколько раз растёт срок до следующего аудита при успехе
THROTTLE_DAYS = 3       # на сколько дней притормаживаем приток новых при срабатывании тормоза


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
    """Выучено: все 4 клетки рампы пройдены (каждая == '1')."""
    return all((modes or {}).get(c, "") == "1" for c in REQUIRED_CELLS)


def _next_step(row, modes):
    """Следующая невыполненная ступень рампы для слова → (step, mode, direction).
    Совсем новое (0 попыток) → пассивная карточка 'card' (study, без направления).
    Иначе — первая клетка REQUIRED_CELLS со значением != '1', из имени выводим mode/direction.
    Если все клетки пройдены (mastered) — возвращаем None."""
    attempts = (row.get("correct") or 0) + (row.get("incorrect") or 0)
    if attempts == 0:
        return ("card", "study", None)
    for cell in REQUIRED_CELLS:
        if (modes or {}).get(cell, "") != "1":
            mode, direction = cell.split("_", 1)
            return (cell, mode, direction)
    return None


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

async def apply_result(user_id: int, pool_id: int, correct: bool, elapsed: float = None,
                       mode: str = None, direction: str = None):
    """Обновить состояние слова после ответа (создаёт строку при первом ответе).
    mode — тип игры (choice/build/input/study/…), direction — направление ('no2int'|'int2no').
    Клетка рампы = f'{mode}_{direction}': верный ответ → '1', ошибка → '' (сброс этой клетки).
    study (карточки) — пассив, в мастери не пишем. build допустим только direction='int2no'.
    direction=None (обратная совместимость) — клетку не трогаем, но hist/счётчики обновляем как раньше.
    «Выучено» = все 4 клетки рампы пройдены (REQUIRED_CELLS)."""
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
        # клетка рампы — только для тестовых типов с заданным направлением (study/без direction — пассив)
        if mode and mode != "study" and direction and direction in _DIR_ALLOWED.get(mode, ()):
            cell = f"{mode}_{direction}"
            if cell in REQUIRED_CELLS:
                modes[cell] = "1" if correct else ""   # верно → пройдена; ошибка → сброс именно этой клетки
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
        # Замыкание петли забывания (§2.4-B, «вариант A»): забытое на аудите слово было
        # де-сертифицировано (certified=0) и доучивалось в общей очереди. Как только оно снова
        # достигает mastered — СРАЗУ ре-сертифицируем, минуя зачётные ворота, и возвращаем под
        # редкий аудит (audit_due = now + FIRST_AUDIT_DAYS). Признак «ранее сертифицировано/выпало
        # из аудита» — was_certified=1; свежие (никогда не сертифицированные) mastered идут через
        # ворота как раньше (§2.4-A), поэтому условие именно certified=0 AND was_certified=1.
        if _mastered_by_modes(modes) and not st.get("certified") and st.get("was_certified"):
            await db.execute(
                "UPDATE user_words SET certified = 1, audit_due = ?, audit_interval = ? "
                "WHERE user_id = ? AND pool_id = ?",
                (_due_str(FIRST_AUDIT_DAYS), float(FIRST_AUDIT_DAYS), user_id, pool_id))
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
            m = {c: "1" for c in REQUIRED_CELLS}   # все клетки рампы пройдены
            m["hist"] = "1" * CAPACITY
            fields = "archived=1, strength=100, reps=MAX(reps,3), modes=?, due_at=?"
            args = (json.dumps(m, ensure_ascii=False), _due_str(120))
        elif action == "reset":
            fields = ("archived=0, strength=0, reps=0, lapses=0, ease=2.5, interval_days=0, "
                      "correct=0, incorrect=0, streak=0, modes=NULL, due_at=NULL, "
                      "certified=0, was_certified=0, audit_due=NULL, audit_interval=0")
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
               uw.correct, uw.incorrect, uw.streak, uw.archived, uw.modes, uw.last_seen,
               uw.certified, uw.audit_due, uw.audit_interval, uw.was_certified
        FROM (
              -- новые слова — только из словарей «в обучении» (studying=1)
              SELECT pool_id FROM dict_words
              WHERE dict_id IN (SELECT id FROM dictionaries
                                WHERE user_id = ? AND COALESCE(studying, 1) = 1)
              UNION
              -- уже начатые (есть прогресс) — остаются в «Учёбе», даже если словарь сняли с обучения
              SELECT dw.pool_id FROM dict_words dw
              JOIN dictionaries d ON d.id = dw.dict_id
              WHERE d.user_id = ?
                AND EXISTS (SELECT 1 FROM user_words uw
                            WHERE uw.user_id = ? AND uw.pool_id = dw.pool_id)
             ) up
        JOIN word_pool wp ON wp.id = up.pool_id
        LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
    """, (user_id, user_id, user_id, user_id)) as cur:
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
        "last_seen": r.get("last_seen"), "certified": bool(r.get("certified")),
        "audit_due": r.get("audit_due"),
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


# ---------------- движок сессии (систему ведёт сервер, режим не выбирает игрок) ----------------

WIP_LIMIT = 20   # лимит слов одновременно-в-работе (new+learning); новые не вводим сверх него


async def build_session(user_id, size=20):
    """Программа занятия, которую готовит СИСТЕМА (без выбора режима игроком).
    Приоритет пулов:
      (1) возвращённые на доучивание / слабые с лапсами (errored, вернулись в учёбу),
      (2) просроченные по due,
      (3) слабые,
      (4) дозревающие (new/learning).
    Для каждого слова берём СЛЕДУЮЩУЮ невыполненную ступень рампы (см. _next_step):
    'card' для совсем нового (0 попыток), иначе первая клетка REQUIRED_CELLS != '1'.
    Лимит одновременно-в-работе: не вводим новые слова, если число не-mastered
    (new+learning) уже >= WIP_LIMIT. Mastered-слова как «новые» не вводим.
    Возвращает [{pool_id, no, translate, mode, direction, step}], не больше size."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        throttled = await _audit_throttled(db, user_id)
    finally:
        await _release(db)
    # держим исходные строки рядом с shape (для статуса/lapses/попыток)
    enriched = []
    for r in rows:
        try:
            modes = json.loads(r.get("modes") or "{}")
        except Exception:
            modes = {}
        st = status_of(r, modes)
        enriched.append({"row": r, "modes": modes, "status": st, "due": _is_due(r)})

    # сколько слов уже в работе (не mastered, не archived, не new) — для лимита притока новых
    in_work = sum(1 for e in enriched if e["status"] in ("learning", "review", "weak"))
    # ворота: пока несданная пачка открыта на экзамен — новые слова не вводим;
    # мягкий тормоз аудита (>THROTTLE забытого) тоже временно замораживает приток новых
    pack_n = sum(1 for e in enriched if e["status"] == "mastered" and not _is_certified(e["row"]))
    had_cert = any(_is_certified(e["row"]) for e in enriched)
    gate_open = pack_n >= (PACK if had_cert else PACK_FIRST) or throttled

    # пулы по приоритету
    def attempts(e):
        rr = e["row"]
        return (rr.get("correct") or 0) + (rr.get("incorrect") or 0)

    returned = sorted(
        [e for e in enriched if e["status"] in ("weak", "learning", "review")
         and (e["row"].get("lapses") or 0) > 0],
        key=lambda e: (e["row"].get("strength") or 0, e["row"].get("due_at") or ""))
    overdue = sorted([e for e in enriched if e["due"]], key=lambda e: e["row"].get("due_at") or "")
    weak = sorted([e for e in enriched if e["status"] == "weak"], key=lambda e: e["row"].get("strength") or 0)
    # дозревающие: всё, что ещё не выучено и не в архиве (new/learning + review, не достигшие mastered)
    maturing = sorted(
        [e for e in enriched if e["status"] in ("new", "learning", "review")],
        key=lambda e: (attempts(e) == 0, e["row"].get("strength") or 0))  # сначала тронутые, новые в хвост

    session, seen = [], set()
    for pool in (returned, overdue, weak, maturing):
        for e in pool:
            pid = e["row"]["pool_id"]
            if pid in seen:
                continue
            # mastered как «новые» не вводим (они и так не попадают в пулы выше)
            if e["status"] in ("mastered", "archived"):
                continue
            # лимит одновременно-в-работе: новое слово (0 попыток) не вводим, если уже >= WIP_LIMIT;
            # а также пока открыты ворота зачётного экзамена / активен мягкий тормоз аудита (приток новых заморожен)
            if attempts(e) == 0 and (in_work >= WIP_LIMIT or gate_open):
                continue
            step = _next_step(e["row"], e["modes"])
            if not step:
                continue
            cell, mode, direction = step
            data = json.loads(e["row"]["data"]) if e["row"].get("data") else {}
            session.append({
                "pool_id": pid, "no": e["row"]["norwegian"],
                "translate": data.get("translate", {}),
                "mode": mode, "direction": direction, "step": cell,
            })
            seen.add(pid)
            if attempts(e) == 0:
                in_work += 1   # ввели новое — слот занят
            if len(session) >= size:
                return {"words": session}
    return {"words": session}


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


def _gate_question(r, lang, distractor_pool):
    """Вопрос-выбор: норвежское слово + 4 варианта перевода (как build_placement).
    distractor_pool — список строк-переводов других слов для дистракторов."""
    import random
    data = json.loads(r["data"]) if r.get("data") else {}
    corr = (data.get("translate", {}).get(lang) or [None])[0]
    if not corr:
        return None
    distract = []
    for t in distractor_pool:
        if t and t != corr and t not in distract:
            distract.append(t)
        if len(distract) == 3:
            break
    if len(distract) < 3:
        return None
    opts = distract + [corr]
    random.shuffle(opts)
    return {"no": r["norwegian"], "pool_id": r["pool_id"], "options": opts}


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
    # дистракторы — переводы любых слов пользователя на нужный язык
    distractor_pool = []
    for r in rows:
        data = json.loads(r["data"]) if r.get("data") else {}
        t = (data.get("translate", {}).get(lang) or [None])[0]
        if t:
            distractor_pool.append(t)
    random.shuffle(pack)
    questions = []
    for r in pack:
        if len(questions) >= SAMPLE:
            break
        random.shuffle(distractor_pool)
        q = _gate_question(r, lang, distractor_pool)
        if q:
            questions.append(q)
    return {"questions": questions, "sample": SAMPLE, "pass": PASS}


def _demote_fields(modes):
    """Демоут слова: сбросить клетки рампы, силу/историю/ease вниз, certified=0.
    Возвращает (modes_json, strength, ease)."""
    m = {k: v for k, v in (modes or {}).items() if k not in REQUIRED_CELLS and k != "hist"}
    m["hist"] = ""
    for c in REQUIRED_CELLS:
        m[c] = ""
    return json.dumps(m, ensure_ascii=False), 0, 1.3


async def _demote(db, user_id, r):
    """Перевести слово mastered→review: сброс клеток рампы, силы/ease вниз, certified=0,
    вернуть в ближайшую ротацию (due завтра, lapses+1, reps→review-уровень)."""
    try:
        modes = json.loads(r.get("modes") or "{}")
    except Exception:
        modes = {}
    modes_json, strength, ease = _demote_fields(modes)
    await db.execute("""
        UPDATE user_words
        SET modes = ?, strength = ?, ease = ?, certified = 0,
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
        # сверка ответов — по правильному переводу слова на язык lang
        correct_n = 0
        missed_pids = []
        for a in (answers or []):
            pid = a.get("pool_id")
            r = by_pid.get(pid)
            if not r:
                continue
            data = json.loads(r["data"]) if r.get("data") else {}
            tr = data.get("translate", {}).get(lang) or []
            ans = (a.get("answer") or "").strip().lower()
            if ans and any(ans == (t or "").strip().lower() for t in tr):
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
    # дистракторы — переводы любых слов пользователя на нужный язык
    distractor_pool = []
    for r in rows:
        data = json.loads(r["data"]) if r.get("data") else {}
        t = (data.get("translate", {}).get(lang) or [None])[0]
        if t:
            distractor_pool.append(t)
    questions = []
    for r in due:
        random.shuffle(distractor_pool)
        q = _gate_question(r, lang, distractor_pool)
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
        checked = refreshed = forgot = 0
        for a in (answers or []):
            r = by_pid.get(a.get("pool_id"))
            if not r:
                continue
            checked += 1
            data = json.loads(r["data"]) if r.get("data") else {}
            tr = data.get("translate", {}).get(lang) or []
            ans = (a.get("answer") or "").strip().lower()
            ok = bool(ans) and any(ans == (t or "").strip().lower() for t in tr)
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


async def learning_stats(user_id):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        activity = await _activity_metrics(db, user_id)
        throttled = await _audit_throttled(db, user_id)
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
    # ворота зачётного экзамена: размер несданной пачки и порог
    pack_n = sum(1 for w in items if w["status"] == "mastered" and not w["certified"])
    had_cert = any(w["certified"] for w in items)
    threshold = PACK if had_cert else PACK_FIRST
    gate = {"pack": pack_n, "threshold": threshold, "open": pack_n >= threshold,
            "toExam": max(0, threshold - pack_n)}
    # аудит забывания: сколько сертифицированных слов уже пора проверить (audit_due <= now)
    now_iso = _now()
    audit_due_n = sum(1 for w in items if w["certified"] and w["audit_due"] and w["audit_due"] <= now_iso)
    # throttled — активен ли мягкий тормоз новых после аудита (>THROTTLE забытого); newBlocked — закрыт ли приток новых вообще
    audit = {"due": audit_due_n, "cap": AUDIT_CAP, "open": audit_due_n > 0, "throttled": throttled}
    return {"total": len(items), "due": due_count, "byStatus": by_status, "byLevel": by_level,
            "currentLevel": current, "toNextLevel": to_next, "startLevel": start, "placed": bool(start),
            "streak": activity["streak"], "today": activity["today"], "accuracy": activity["accuracy"],
            "retention": retention, "masteredWeek": mastered_week, "gate": gate, "audit": audit}


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
    from .pool import get_pool_duel_words
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
            if is_input:
                # продукция: показываем перевод (prompt), ключ грейда — норвежская лемма (no)
                questions.append({"no": w["norwegian"], "prompt": corr, "level": lv, "type": "input"})
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
            questions.append({"no": w["norwegian"], "level": lv, "type": "choice", "options": opts})
            picked += 1
    random.shuffle(questions)
    return {"questions": questions}


# Калибровка входного теста — консервативная (тест НЕ должен завышать уровень):
PLACEMENT_PASS = 0.8     # порог сдачи уровня: >= 80% верных
PLACEMENT_MIN = 4        # минимум отвеченных вопросов на уровне, иначе уровень не «сдан»


async def grade_placement(user_id, lang, answers):
    """answers: [{no, level, answer, type}]. Оцениваем долю верных по уровням и КОНСЕРВАТИВНО
    оцениваем стартовый уровень: идём снизу вверх и засчитываем уровень только если на нём
    отвечено достаточно (>= PLACEMENT_MIN) и доля верных >= PLACEMENT_PASS; останавливаемся
    на ПЕРВОМ уровне со сдачей ниже порога (округление вниз). Проверка ответов — по пулу.
    Сверка по типу вопроса:
      • type=="choice": верно если answer ∈ переводам слова no на язык lang (узнавание);
      • type=="input": верно если введённое совпало с НОРВЕЖСКОЙ леммой no (или любым из
        translate.no) СНИСХОДИТЕЛЬНО — через _fold_loose (lower/trim, å→a ø→o æ→ae, срез диакритик).
    Тип берём из ответа; если его нет — выводим из уровня (верхние → input), чтобы не падать на
    старых клиентах."""
    from .pool import get_pool_id, get_pool_by_id
    per_lvl = {lv: {"ok": 0, "total": 0} for lv in LEVELS}
    for a in (answers or []):
        lv = a.get("level")
        if lv not in per_lvl:
            continue
        per_lvl[lv]["total"] += 1
        no = a.get("no") or ""
        pid = await get_pool_id(no)
        if not pid:
            continue
        p = await get_pool_by_id(pid)
        translate = ((p or {}).get("data") or {}).get("translate", {})
        qtype = a.get("type") or ("input" if lv in PLACEMENT_INPUT_LEVELS else "choice")
        if qtype == "input":
            # продукция: введённое сравниваем с норвежской леммой no и всеми вариантами translate.no
            ans = _fold_loose(a.get("answer") or "")
            keys = {_fold_loose(no)} | {_fold_loose(t) for t in (translate.get("no") or [])}
            keys.discard("")
            if ans and ans in keys:
                per_lvl[lv]["ok"] += 1
        else:
            tr = translate.get(lang) or []
            ans = (a.get("answer") or "").strip().lower()
            if ans and any(ans == (t or "").strip().lower() for t in tr):
                per_lvl[lv]["ok"] += 1
    # уровень = высший пройденный подряд снизу; стоп на первом несданном (консервативно, вниз)
    level = "A1"
    for lv in LEVELS:
        d = per_lvl[lv]
        passed = d["total"] >= PLACEMENT_MIN and (d["ok"] / d["total"]) >= PLACEMENT_PASS
        if passed:
            level = lv
        else:
            break
    await set_start_level(user_id, level)
    return {"level": level, "perLevel": per_lvl}


STARTER_GOAL = 20   # сколько слов гарантируем новичку после калибровки, чтобы было что учить


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
    res = await suggest_words(user_id, count=need, level=level)
    return {"seeded": res.get("added", 0), "had": have}


async def estimate_level(user_id):
    """Рабочий уровень для подсказок — текущий уровень по целям (первый незакрытый)."""
    return (await learning_stats(user_id))["currentLevel"]


async def suggest_words(user_id, count=10, level=None):
    """«Докинуть слов»: добавить НОВЫЕ слова пула по уровню пользователя, которых у него ещё нет,
    в СКРЫТЫЙ авто-словарь (hidden=1, studying=1) — чтобы не засорять личные словари, но они
    были видны в Учёбе. Возвращает добавленные. Импорт здесь, чтобы избежать циклов.
    Гейт ворот: пока несданная пачка открыта на экзамен — приток новых слов закрыт."""
    from .pool import get_pool_duel_words, get_pool_id
    from .dictionaries import add_word_to_dict, get_or_create_hidden_dict
    if await new_words_blocked(user_id):
        return {"added": 0, "words": [], "level": None, "blocked": True}
    lvl = level if level in LEVELS else await estimate_level(user_id)
    # целевой — скрытый авто-словарь (получить-или-создать)
    target_id = await get_or_create_hidden_dict(user_id)
    db = await _conn()
    try:
        # уже имеющиеся слова — по ВСЕМ словарям пользователя (чтобы не дублировать существующие)
        async with db.execute("SELECT DISTINCT pool_id FROM dict_words WHERE dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)", (user_id,)) as cur:
            have = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    # кандидаты по уровню, с запасом, исключаем уже имеющиеся
    cand = await get_pool_duel_words(max(count * 4, 40), lvl, None)
    added = []
    for w in cand:
        if len(added) >= count:
            break
        pid = await get_pool_id(w["norwegian"])
        if not pid or pid in have:
            continue
        res = await add_word_to_dict(user_id, target_id, pid)
        if res.get("id") and not res.get("duplicate"):
            added.append({"no": w["norwegian"], "translate": w.get("translate", {})})
            have.add(pid)
    return {"added": len(added), "words": added, "level": lvl, "dict": "__auto__"}
