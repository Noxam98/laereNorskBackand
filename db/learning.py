"""«Учёба» — слой интервальных повторений (SRS) над всеми словами пользователя.
Слова берутся из словарей (dict_words → word_pool), состояние — в user_words.
Статусы вычисляются на чтении из силы/попыток; archived — ручной флаг («я это знаю»)."""
import asyncio
import json
import unicodedata
from datetime import datetime, timedelta
import fuzzy
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
# Рампа служебных слов (A1): карточка → 3 разных cloze (направление = индекс предложения 1..3).
FUNC_CELLS = ["cloze_1", "cloze_2", "cloze_3"]
ALL_CELLS = REQUIRED_CELLS + FUNC_CELLS
# build (собери из букв) осмыслен только в направлении родной→норв; cloze — индекс предложения
_DIR_ALLOWED = {"choice": ("no2int", "int2no"), "build": ("int2no",), "input": ("no2int", "int2no"),
                "cloze": ("1", "2", "3")}
# Прогресс — скользящее окно: новые попытки вытесняют старые (ёмкость). Сила слова считается
# по последним CAPACITY попыткам, а не за всю историю — старые успехи «выпадают» со временем.
CAPACITY = 8
DAILY_GOAL = 20   # дневная цель по умолчанию (слов/ответов)
COOLDOWN_MIN = 12  # «умная очередь»: только что показанное слово не ставим в начало следующей сессии (мин)
FUNC_GATE = 20     # служебные слова не вводим новыми, пока контентных (learning+review+mastered) < этого

# --- Класс «служебное слово» (A1): союз/предлог/местоимение/детерминатив + частица å + функц. наречия ---
_FUNC_CORE_POS = {"konjunksjon", "subjunksjon", "preposisjon", "pronomen", "determinativ"}
_FUNC_ADV_WL = {"ikke", "inn", "ut", "opp", "ned", "her", "der", "nå", "da", "også", "hjem",
                "hit", "dit", "fram", "frem", "bort", "hjemme", "ute", "inne", "oppe", "nede",
                "borte", "alltid", "aldri", "kanskje"}


def is_function_word(norwegian, data):
    """Служебное слово: ядро по POS (союз/предлог/местоимение/детерминатив), частица «å»,
    или функциональное наречие из белого списка. data — распарсенный dict из word_pool."""
    no = (norwegian or "").strip().lower()
    if no == "å":
        return True
    pos = ((data or {}).get("part_of_speech") or "").strip().lower()
    if pos in _FUNC_CORE_POS:
        return True
    if pos == "adverb" and no in _FUNC_ADV_WL:
        return True
    return False


# Пословный «словарный порог» для служебных слов: служебное слово вводим в обучение (cloze) только
# когда выученных КОНТЕНТНЫХ слов ≥ N — иначе не из чего строить осмысленное предложение. Для «og»
# хватает 2 любых слов, для «istedenfor» нужна пара контрастных понятий → выше. Заполняется
# калибровкой (см. cloze-vocab-thresholds). Для слова не из карты — дефолт FUNC_GATE.
CLOZE_MIN = {
    # базовый «клей» — работает почти с любыми 1–2 словами
    "å": 4, "jeg": 4, "og": 4, "i": 5, "ikke": 5, "der": 5, "hjem": 5, "hjemme": 6,
    "ute": 7, "oppe": 8, "inne": 8, "kanskje": 8, "noe": 9, "noen": 9,
    # направления/местоимения/частотные — нужен 1 предмет/место/предикат
    "opp": 10, "ned": 10, "inn": 10, "frem": 10, "dem": 10,
    "alltid": 11, "min": 11, "bort": 11, "denne": 11, "til": 11, "også": 11,
    "gjennom": 12, "hver": 12, "aldri": 12, "deres": 12, "over": 13, "under": 13,
    # нужны 2 ориентира / тема / последовательность
    "om": 14, "foran": 14, "etter": 14, "hos": 14, "langs": 15, "utenfor": 15,
    "begge": 16, "uten": 16, "selv": 16, "av": 16, "mellom": 17, "hvilken": 18,
    # двухситуативные / контраст / абстракция — нужно несколько разных понятий
    "hvis": 20, "imot": 20, "men": 20, "både": 22, "fordi": 22, "mens": 22,
    "ettersom": 24, "ifølge": 24, "hverken": 26, "istedenfor": 28,
}  # {norwegian_lower: N}; калибровка cloze-vocab-thresholds (2 агента, ревью)


def _cloze_min(norwegian):
    """Сколько выученных контентных слов нужно, прежде чем вводить это служебное слово."""
    return CLOZE_MIN.get((norwegian or "").strip().lower(), FUNC_GATE)


def required_cells(row):
    """Клетки рампы для слова: служебные → 3 cloze; остальные → choice×2/build/input.
    row — строка с norwegian + data (json-строка или dict)."""
    try:
        d = row.get("data")
        d = json.loads(d) if isinstance(d, str) else (d or {})
    except Exception:
        d = {}
    return FUNC_CELLS if is_function_word(row.get("norwegian"), d) else REQUIRED_CELLS


def _is_mastered(row, modes):
    """Выучено: все клетки рампы слова == '1'. Grandfathering: служебное, выученное СТАРОЙ рампой
    (choice/build/input до перехода на cloze), считаем выученным — прогресс не сбрасываем."""
    m = modes or {}
    cells = required_cells(row)
    if all(m.get(c, "") == "1" for c in cells):
        return True
    if cells is FUNC_CELLS and all(m.get(c, "") == "1" for c in REQUIRED_CELLS):
        return True
    return False
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
    if _is_mastered(row, modes):   # в т.ч. grandfathered служебные (не гоняем по cloze повторно)
        return None
    for cell in required_cells(row):
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
    if _is_mastered(row, modes):
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
        # класс слова (служебное → cloze-клетки) для корректной рампы/мастери
        async with db.execute("SELECT norwegian, data FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            wp = await cur.fetchone()
        wrow = {"norwegian": (wp["norwegian"] if wp else None), "data": (wp["data"] if wp else None)}
        cells = required_cells(wrow)
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
            if cell in cells:
                if correct:
                    modes[cell] = "1"              # верно → ступень пройдена
                else:
                    modes[cell] = ""               # ошибка → текущая ступень сброшена
                    i = cells.index(cell)
                    if i > 0:                      # ОТКАТ на одну ступень назад (не ниже первой клетки —
                        modes[cells[i - 1]] = ""   # т.е. карточку-интро откат никогда не трогает)
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
        if _is_mastered(wrow, modes) and not st.get("certified") and st.get("was_certified"):
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
        return {"ok": True, "strength": strength, "due_at": due, "modes": modes, "mastered": _is_mastered(wrow, modes)}
    finally:
        await _release(db)


async def set_status(user_id: int, pool_id: int, action: str):
    """action: know (в архив, сила 100) | reset (сброс) | unarchive (вернуть в ротацию)."""
    db = await _conn()
    try:
        if action == "know":
            m = {c: "1" for c in ALL_CELLS}   # все клетки рампы пройдены (и choice/build/input, и cloze)
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


async def learning_add(user_id: int, pool_id: int):
    """Добавить слово из Базы прямо в Учёбу — кладём в скрытый авто-словарь (studying=1),
    чтобы не плодить именованные словари. Идемпотентно (UNIQUE dict_id,pool_id).
    Если слово раньше «мягко удаляли» (архив с сохранённым прогрессом) — возвращаем в ротацию
    (archived=0), т.е. прогресс восстанавливается. Иначе слово появляется как «новое»."""
    from .dictionaries import add_word_to_dict, get_or_create_hidden_dict
    dict_id = await get_or_create_hidden_dict(user_id)
    res = await add_word_to_dict(user_id, dict_id, pool_id)
    if res.get("error"):
        return res
    db = await _conn()
    try:
        await db.execute(
            "UPDATE user_words SET archived = 0, due_at = ? WHERE user_id = ? AND pool_id = ? AND archived = 1",
            (_due_str(1), user_id, pool_id))
        await db.commit()
    finally:
        await _release(db)
    return {"ok": True, "pool_id": pool_id, "duplicate": bool(res.get("duplicate"))}


async def learning_remove(user_id: int, pool_id: int):
    """Мягко убрать слово из Учёбы: отвязать от словарей пользователя (dict_words), а прогресс
    (user_words), если он есть, НЕ удалять — архивировать (archived=1). Так слово уходит из
    ротации и из всех списков/счётчиков Учёбы (не попадает в _fetch_user_words без dict_words),
    но при повторном добавлении прогресс восстановится. Свежее «новое» (без user_words) — просто
    отвязка (чистая отмена добавления)."""
    db = await _conn()
    try:
        await db.execute(
            "DELETE FROM dict_words WHERE pool_id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)",
            (pool_id, user_id))
        await db.execute("UPDATE user_words SET archived = 1 WHERE user_id = ? AND pool_id = ?", (user_id, pool_id))
        await db.commit()
        return {"ok": True}
    finally:
        await _release(db)


# ---------------- выборка слов пользователя ----------------

async def _fetch_user_words(db, user_id):
    """Все слова пользователя (уникальные по пулу) + состояние SRS + темы."""
    async with db.execute("""
        SELECT wp.id AS pool_id, wp.norwegian, wp.data, wp.level, wp.freq, wp.forms, (wp.tts IS NOT NULL) AS has_tts,
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
        "level": r.get("level"), "freq": r.get("freq"), "topics": r.get("topics", []), "hasTts": bool(r.get("has_tts")),
        "status": status_of(r, modes), "strength": r.get("strength") or 0, "due_at": r.get("due_at"),
        "modes": modes, "correct": r.get("correct") or 0, "incorrect": r.get("incorrect") or 0, "due": _is_due(r),
        "last_seen": r.get("last_seen"), "certified": bool(r.get("certified")),
        "audit_due": r.get("audit_due"),
    }


async def get_learning(user_id, status=None, level=None, topic=None, q=None, sort="strength", order="asc", limit=200, offset=0):
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
    # Единые типы сортировки (как в Пуле/Словаре): базово по возрастанию, направление — order.
    _no = lambda w: (w["no"] or "").lower()
    _LR = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}
    _keys = {
        "alpha":    _no,
        "due":      lambda w: ((w["due_at"] or "9999"), _no(w)),
        "level":    lambda w: (_LR.get(w["level"] or "", 99), _no(w)),
        "freq":     lambda w: (w["freq"] if w.get("freq") is not None else -1, _no(w)),
        "pos":      lambda w: ((w["part_of_speech"] or "￿"), _no(w)),
        "strength": lambda w: (w["strength"], _no(w)),
    }
    items.sort(key=_keys.get(sort, _keys["strength"]))
    if str(order).lower() == "desc":
        items.reverse()
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


async def _attach_choice_options(session, lang, n=3):
    """Для choice-элементов сессии дотягиваем варианты ответа (как /pool/{id}/distractors),
    чтобы фронт получил ВСЁ нужное одним запросом и не догружал ничего во время сессии.
    Мутирует элементы: добавляет options=[{w,alt}] и distractors=[w]. Семантически близкие
    по эмбеддингам (ANN). lang — язык переводов-вариантов, direction — у каждого элемента."""
    choice = [e for e in session if e.get("mode") == "choice"]
    if not choice:
        return
    from llm import ranked_pool  # ленивый импорт — избегаем цикла импортов на старте
    # эмбеддинги нужных слов — одним запросом
    pids = [e["pool_id"] for e in choice]
    emb = {}
    db = await _conn()
    try:
        marks = ",".join("?" for _ in pids)
        async with db.execute(f"SELECT id, embedding FROM word_pool WHERE id IN ({marks})", pids) as cur:
            for r in await cur.fetchall():
                emb[r["id"]] = r["embedding"]
    finally:
        await _release(db)

    def answer_of(d, no, direction):
        # до ДВУХ вариантов ответа: (основной, второй|None)
        if direction == "int2no":
            return (no, None)
        tr = ((d or {}).get("translate", {}) or {}).get(lang) or []
        return (tr[0] if tr else None, tr[1] if len(tr) > 1 else None)

    for e in choice:
        direction = e.get("direction") or "int2no"
        no = e.get("no") or ""
        tr_all = e.get("translate", {}) or {}
        data = {"translate": tr_all}
        # СМЫСЛ цели = все её переводы на язык юзера. Дистрактор-СИНОНИМ (его переводы пересекаются
        # со смыслом цели) исключаем — иначе вариант оказался бы тоже верным и вопрос нечестным
        # (avstand=[расстояние,дистанция] vs distanse=[дистанция,расстояние]).
        target_mean = {x.strip().lower() for x in (tr_all.get(lang) or []) if x}
        # плюс не повторяем сами допустимые ответы (все переводы / норв. формы цели)
        own = ({(no or "").strip().lower()} | {x.strip().lower() for x in (tr_all.get("no") or []) if x}) \
            if direction == "int2no" else set(target_mean)
        own.discard("")
        ordered = await ranked_pool(emb.get(e["pool_id"]), no, 40) if emb.get(e["pool_id"]) else []
        out, seen = [], set(own)
        for c in ordered:
            cd = c.get("data") or {}
            cmean = {x.strip().lower() for x in ((cd.get("translate", {}) or {}).get(lang) or []) if x}
            if target_mean and (cmean & target_mean):   # синоним по смыслу — не годится в дистракторы
                continue
            a, alt = answer_of(cd, c.get("norwegian"), direction)
            la = (a or "").strip().lower()
            if a and la not in seen:
                out.append({"w": a, "alt": alt}); seen.add(la)
            if len(out) >= n:
                break
        e["options"] = out
        e["distractors"] = [o["w"] for o in out]


async def build_session(user_id, size=20, lang="ru"):
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
    async def _load():
        """Прочитать слова пользователя + флаг тормоза, разложить по статусам."""
        db = await _conn()
        try:
            rows = await _fetch_user_words(db, user_id)
            throttled = await _audit_throttled(db, user_id)
        finally:
            await _release(db)
        enriched = []
        for r in rows:
            try:
                modes = json.loads(r.get("modes") or "{}")
            except Exception:
                modes = {}
            st = status_of(r, modes)
            try:
                pdata = json.loads(r["data"]) if r.get("data") else {}
            except Exception:
                pdata = {}
            enriched.append({"row": r, "modes": modes, "status": st, "due": _is_due(r), "data": pdata})
        # сколько слов уже в работе (не mastered, не archived, не new) — для лимита притока новых
        in_work = sum(1 for e in enriched if e["status"] in ("learning", "review", "weak"))
        # ворота: пока несданная пачка открыта на экзамен — новые слова не вводим;
        # мягкий тормоз аудита (>THROTTLE забытого) тоже временно замораживает приток новых
        pack_n = sum(1 for e in enriched if e["status"] == "mastered" and not _is_certified(e["row"]))
        had_cert = any(_is_certified(e["row"]) for e in enriched)
        gate_open = pack_n >= (PACK if had_cert else PACK_FIRST) or throttled
        return enriched, in_work, gate_open

    enriched, in_work, gate_open = await _load()

    # ГЕЙТ A1 (#1,#2): база контентных слов (learning/review/mastered, НЕ служебные). Пока её нет —
    # служебные слова не вводим и не досыпаем (есть из чего строить cloze только после базы).
    def _content_known(items):
        return sum(1 for e in items
                   if e["status"] in ("learning", "review", "mastered")
                   and not is_function_word(e["row"]["norwegian"], e["data"]))
    content_known = _content_known(enriched)
    func_gate_ok = content_known >= FUNC_GATE   # грубый общий гейт (для авто-добора служебных)

    def _func_locked(e):
        """Новое служебное слово ещё рано вводить: выученных контентных < его пословного порога."""
        return (is_function_word(e["row"]["norwegian"], e["data"])
                and content_known < _cloze_min(e["row"]["norwegian"]))

    # АВТО-ДОБОР: «Учёба» не должна пустеть сама собой, но и не навязывает «Базу».
    # Семантика — «закрыть ПУСТОТУ, когда пул НОВЫХ иссяк», а НЕ «добить до WIP_LIMIT».
    # Доступные новые (status == new) — это «пул новых». Досыпаем ТОЛЬКО когда он пуст
    # (new == 0): пока у юзера есть хоть одно своё новое слово, добор не срабатывает и
    # «База» не подмешивается. Дополнительно: ворота не закрыты (не ждём экзамен / нет
    # тормоза аудита) И есть свободный слот в работе (in_work < WIP_LIMIT — иначе новые
    # всё равно не вводятся по лимиту). Сколько досыпать — до WIP_LIMIT по in_work.
    if not gate_open:
        # залоченные пословным порогом новые служебные не считаем «доступными новыми» — иначе они
        # держат new_avail>0 и блокируют добор контентных, которыми сами же и разблокируются
        new_avail = sum(1 for e in enriched if e["status"] == "new" and not _func_locked(e))
        if new_avail == 0 and in_work < WIP_LIMIT:
            level = await estimate_level(user_id)
            res = await suggest_words(user_id, count=WIP_LIMIT - in_work, level=level, allow_func=func_gate_ok)
            if res.get("added"):
                # перечитываем слова после добора и продолжаем обычную сборку
                enriched, in_work, gate_open = await _load()
                content_known = _content_known(enriched)
                func_gate_ok = content_known >= FUNC_GATE

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

    # «Умная очередь»: слово, показанное ТОЛЬКО ЧТО (last_seen свежее кулдауна), не ставим в начало
    # следующей сессии — откладываем в ХВОСТ очереди (а при достатке слов — фактически в следующую
    # сессию). Так только что подсказанное слово не всплывает сразу же; через кулдаун вернётся штатно.
    cooldown_cut = (datetime.utcnow() - timedelta(minutes=COOLDOWN_MIN)).isoformat()
    def is_fresh(e):
        ls = e["row"].get("last_seen")
        return bool(ls) and ls >= cooldown_cut   # last_seen — UTC ISO, сравнение строк = по времени

    # func_gate_ok уже посчитан выше (после загрузки/добора)
    # кандидаты в порядке приоритета (дедуп; без mastered/archived и без следующего шага рампы)
    cand, seen, seen_wp = [], set(), set()
    for pool in (returned, overdue, weak, maturing):
        for e in pool:
            pid = e["row"]["pool_id"]
            # не допускаем в сессии два одинаковых слова с одинаковой частью речи (омонимы с
            # РАЗНЫМ pos — можно, это разные слова; одинаковые (norwegian, pos) — нет)
            wp = (e["row"]["norwegian"], (e["data"] or {}).get("part_of_speech", "") or "")
            if pid in seen or wp in seen_wp or e["status"] in ("mastered", "archived"):
                continue
            # новое служебное слово придерживаем, пока выученных контентных < его пословного порога
            if attempts(e) == 0 and _func_locked(e):
                continue
            step = _next_step(e["row"], e["modes"])
            if not step:
                continue
            seen.add(pid)
            seen_wp.add(wp)
            cand.append((e, step))

    # недавно показанные — в хвост (не лидируют следующую сессию); остальные сохраняют приоритет
    ordered = [c for c in cand if not is_fresh(c[0])] + [c for c in cand if is_fresh(c[0])]

    # cloze-кэш для служебных слов на стадии cloze (предложения готовим заранее, фоном)
    cloze_pids = [e["row"]["pool_id"] for (e, step) in ordered if step[1] == "cloze"]
    cloze_map = {}
    if cloze_pids:
        dbc = await _conn()
        try:
            cloze_map = await get_cloze_map(dbc, user_id, cloze_pids)
        finally:
            await _release(dbc)

    session = []
    for e, step in ordered:
        # лимит притока новых: новое слово (0 попыток) не вводим при заполненном WIP / открытых воротах
        if attempts(e) == 0 and (in_work >= WIP_LIMIT or gate_open):
            continue
        cell, mode, direction = step
        pid = e["row"]["pool_id"]
        data = json.loads(e["row"]["data"]) if e["row"].get("data") else {}
        el = {
            "pool_id": pid, "no": e["row"]["norwegian"],
            "translate": data.get("translate", {}),
            "part_of_speech": data.get("part_of_speech", ""),
            "gloss": data.get("gloss"), "example": data.get("example"),  # для карточки служебного (Ф2)
            "forms": (json.loads(e["row"]["forms"]) if e["row"].get("forms") else None),  # колонка wp.forms (не data!) — для артикля (en/ei/et) сущ. и «å» глаг.
            "mode": mode, "direction": direction, "step": cell,
        }
        if mode == "cloze":
            items = cloze_map.get(pid)
            idx = (int(direction) - 1) if str(direction).isdigit() else 0
            if not items or idx >= len(items):
                # cloze ещё не сгенерён — запускаем фоном, это слово сейчас пропускаем (придёт позже)
                asyncio.create_task(generate_cloze(user_id, pid))
                continue
            el["cloze"] = items[idx]
        session.append(el)
        if attempts(e) == 0:
            in_work += 1   # ввели новое — слот занят
        if len(session) >= size:
            break
    await _attach_choice_options(session, lang)
    return {"words": session}


# ---------------- cloze для служебных слов (A1, Ф4) ----------------
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


def _demote_fields(modes):
    """Демоут слова: сбросить клетки рампы, силу/историю/ease вниз, certified=0.
    Возвращает (modes_json, strength, ease)."""
    m = {k: v for k, v in (modes or {}).items() if k not in ALL_CELLS and k != "hist"}
    m["hist"] = ""
    for c in ALL_CELLS:
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
        known = await _known_vocab(db)
        # сверка ответов — нечётко, по любой «своей» форме слова (перевод/норв./словоформы)
        correct_n = 0
        missed_pids = []
        for a in (answers or []):
            pid = a.get("pool_id")
            r = by_pid.get(pid)
            if not r:
                continue
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
        for a in (answers or []):
            r = by_pid.get(a.get("pool_id"))
            if not r:
                continue
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


async def suggest_words(user_id, count=10, level=None, allow_func=True):
    """«Докинуть слов»: добавить НОВЫЕ слова пула по уровню пользователя, которых у него ещё нет,
    в СКРЫТЫЙ авто-словарь (hidden=1, studying=1) — чтобы не засорять личные словари, но они
    были видны в Учёбе. Возвращает добавленные. Импорт здесь, чтобы избежать циклов.
    Гейт ворот: пока несданная пачка открыта на экзамен — приток новых слов закрыт."""
    from .pool import pool_by_freq, pool_by_freq_topics
    from .users import get_user_focus_topics
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
    # кандидаты по уровню, по ЧАСТОТНОСТИ. ВАЖНО: окно расширяем за все уже имеющиеся слова —
    # иначе у юзера с большим словарём топ-N по частоте уже целиком его, новых не находится и пул
    # кажется «исчерпанным» (сессии тают до 1–3 слов, перестают смешиваться).
    window = len(have) + max(count * 6, 60)
    cand = await pool_by_freq(window, lvl)
    # Фокус на темах: ~1 из 3 кандидатов — из выбранных тем (по частоте), пока тема-слова не кончатся.
    # Пусто → cand без изменений (поведение ровно как раньше). Дедуп — общим циклом ниже (have).
    focus = await get_user_focus_topics(user_id)
    if focus:
        topic_cand = await pool_by_freq_topics(window, lvl, focus)
        if topic_cand:
            merged, ti, ni = [], 0, 0
            while ti < len(topic_cand) or ni < len(cand):
                if ti < len(topic_cand):
                    merged.append(topic_cand[ti]); ti += 1
                for _ in range(2):
                    if ni < len(cand):
                        merged.append(cand[ni]); ni += 1
            cand = merged
    added = []
    for w in cand:
        if len(added) >= count:
            break
        pid = w["pool_id"]
        if not pid or pid in have:
            continue
        # гейт A1: пока нет базы контентных — служебные не досыпаем (иначе «новый пул» забьётся ими)
        if not allow_func and is_function_word(w["norwegian"], {"part_of_speech": w.get("part_of_speech")}):
            continue
        res = await add_word_to_dict(user_id, target_id, pid)
        if res.get("id") and not res.get("duplicate"):
            added.append({"no": w["norwegian"], "translate": w.get("translate", {})})
            have.add(pid)
    return {"added": len(added), "words": added, "level": lvl, "dict": "__auto__"}
