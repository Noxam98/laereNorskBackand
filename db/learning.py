"""«Учёба» — слой интервальных повторений (SRS) над всеми словами пользователя.
Слова берутся из словарей (dict_words → word_pool), состояние — в user_words.
Статусы вычисляются на чтении из силы/попыток; archived — ручной флаг («я это знаю»)."""
import asyncio
import json
import os
import time
import unicodedata
from datetime import datetime, timedelta
import fuzzy
from config import logger
from .core import _conn, _release, _now
from .pool_queues import has_tts_expr   # единый источник правды «есть озвучка» (Этап 1)

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
# Порядок = порядок прохождения (_next_step берёт первую непройденную). Аудио-вопрос (choice_no2int,
# «на слух») стоит ВТОРЫМ. Поведение зависит от настройки аудиозаданий (gamePrefs.audio):
#   audio ВКЛ  — _next_step ПРОПУСКАЕТ choice_no2int (текстовые ступени идут вперёд), а аудио-клетка
#                откладывается в отдельную слуховую партию (build_listen_session): слово «выучено»
#                только после сдачи на слух;
#   audio ВЫКЛ — choice_no2int идёт штатно на 2-м месте как ТЕКСТ (норв.→выбор перевода), слуховой
#                сессии нет, слово выучивается полностью текстом.
# Канонические наборы клеток живут в ЧИСТОМ ядре srs/ (Этап 2 decoupling-плана);
# здесь — реэкспорт для обратной совместимости импортов (tests, exams, роутеры).
from srs.cells import (REQUIRED_CELLS, FUNC_CELLS, FUNC_CELLS_CHOICE, PHRASE_CELLS,
                       ALL_CELLS, AUDIO_CELL as _AUDIO_CELL)
from srs import cells as _srs_cells, status as _srs_status, steps as _srs_steps
from srs import gates as _gates
from session import (pools as _pools, forms_phase as _forms_phase,
                     distractors as _distractors, reason as _reason)
from session.shape import make_element as _make_element
# Тип задания «вставь пропущенное» (cloze) для служебных слов можно временно выключить
# (CLOZE_ENABLED=0, дефолт — ВЫКЛ). Тогда служебные слова идут упрощённой рампой «только выбор»
# (карточка → choice×2 → выучено), без cloze. Вернуть — Fly secret CLOZE_ENABLED=1.
CLOZE_ENABLED = os.getenv("CLOZE_ENABLED", "0") == "1"
# (наборы клеток FUNC_CELLS_CHOICE / PHRASE_CELLS реэкспортированы из srs.cells выше)
# Грамматические клетки (overlay-тир ★): НЕ входят в base-рампу и в «выучено»/CEFR — отдельный слой
# ПОВЕРХ уже выученных слов, проверяющий формы. Доля сессии × частотный приоритет, гейтится тумблером
# профиля (gamePrefs.grammar). NOUN-срез (существительное):
#   choice_gender — выбрать артикль (en/ei/et) к лемме (узнавание рода);
#   input_indefpl — напечатать неопр. мн.ч. (ТОЛЬКО если оно нерегулярно — иначе выводится правилом).
NOUN_GRAMMAR_CELLS = ["choice_gender", "input_indefpl"]
# VERB-срез: ввод нерегулярной формы (сильные/супплетивные — не выводятся слабым правилом).
VERB_GRAMMAR_CELLS = ["input_past", "input_perfect", "input_present"]
# ADJECTIVE-срез: согласование/степени, когда факт ≠ предсказанию правила (или супплетив).
ADJ_GRAMMAR_CELLS = ["input_neuter", "input_comparative", "input_superlative", "input_pluraladj"]
# PRONOUN/DETERMINER-срез (КУРИРУЕМЫЙ закрытый класс — не выводится правилом, нет в forms_loop):
# падеж личных местоимений (субъект→объект) + согласование притяжательных. Формы сидятся в
# word_pool.forms на старте (seed_pronoun_forms), дальше — обычный overlay.
PRONOUN_GRAMMAR_CELLS = ["input_objcase", "input_possneut", "input_posspl"]
PRONOUN_PARADIGM = {
    "jeg": {"obj": "meg"}, "du": {"obj": "deg"}, "han": {"obj": "ham"},
    "hun": {"obj": "henne"}, "vi": {"obj": "oss"}, "de": {"obj": "dem"},
    "min": {"neuter": "mitt", "plural": "mine"}, "din": {"neuter": "ditt", "plural": "dine"},
    "sin": {"neuter": "sitt", "plural": "sine"}, "vår": {"neuter": "vårt", "plural": "våre"},
}
# Таблица input-клеток форм: cell → (поле forms с верным ответом, ключ подписи формы для FormPrompt).
# Все они mode=input, direction = cell.split('_',1)[1] (см. _grammar_element / _DIR_ALLOWED).
_INPUT_FORM_CELLS = {
    "input_indefpl":     ("indef_pl",    "indef_pl"),     # сущ.: неопр. мн.ч.
    "input_present":     ("present",     "present"),       # глаг.: презенс
    "input_past":        ("past",        "past"),          # глаг.: претерит
    "input_perfect":     ("perfect",     "perfect"),       # глаг.: перфект (полная форма с aux)
    "input_neuter":      ("neuter",      "neuter"),        # прил.: ср. род
    "input_comparative": ("comparative", "comparative"),   # прил.: сравнит.
    "input_superlative": ("superlative", "superlative"),   # прил.: превосх.
    "input_pluraladj":   ("plural",      "plural_adj"),     # прил.: мн./опр. форма
    "input_objcase":     ("obj",         "objcase"),       # местоим.: объектный падеж (jeg→meg)
    "input_possneut":    ("neuter",      "poss_neuter"),    # притяж.: ср. род (min→mitt)
    "input_posspl":      ("plural",      "poss_plural"),    # притяж.: мн. (min→mine)
}
GRAMMAR_RATIO = 0.3   # доля грамм-упражнений от размера сессии: round(size*RATIO) (см. build_session, D4)
_GENDERS = ("en", "ei", "et")
# Группа пер-POS тумблера грамматики (gamePrefs.grammarPos): местоим.+притяж. — под одной группой.
_GRAMMAR_GROUP = {"noun": "noun", "verb": "verb", "adjective": "adjective",
                  "pronoun": "pronoun", "determiner": "pronoun"}
# build (собери из букв) осмыслен только в направлении родной→норв; cloze — индекс предложения;
# order (порядок слов) / cells (буквенные клетки фразы) — только родная→норв (продукция);
# gender — выбор артикля сущ.; indefpl — ввод неопр. мн.ч. (грамм-overlay)
_DIR_ALLOWED = {"choice": ("no2int", "int2no", "gender"), "build": ("int2no",),
                "input": ("no2int", "int2no", "indefpl",                          # сущ.
                          "present", "past", "perfect",                           # глаг.
                          "neuter", "comparative", "superlative", "pluraladj",     # прил.
                          "objcase", "possneut", "posspl"),                        # местоим./притяж.
                "cloze": ("1", "2", "3"), "order": ("int2no",), "cells": ("int2no",)}
# Прогресс — скользящее окно: новые попытки вытесняют старые (ёмкость). Сила слова считается
# по последним CAPACITY попыткам, а не за всю историю — старые успехи «выпадают» со временем.
CAPACITY = 8
DAILY_GOAL = 20   # дневная цель по умолчанию (слов/ответов)
COOLDOWN_MIN = 12  # «умная очередь»: только что показанное слово не ставим в начало следующей сессии (мин)
FUNC_GATE = 20     # служебные слова не вводим новыми, пока контентных (learning+review+mastered) < этого

# --- Класс «служебное слово» (A1): союз/предлог/местоимение/детерминатив + частица å + функц. наречия ---
# normalize_pos сводит норв.↔англ. написания к канону (substantiv→noun, preposisjon→preposition…),
# поэтому набор служебных — в каноничных англ. ключах (см. pos.FUNCTION_POS).
from pos import normalize_pos, FUNCTION_POS as _FUNC_CORE_POS
_FUNC_ADV_WL = {"ikke", "inn", "ut", "opp", "ned", "her", "der", "nå", "da", "også", "hjem",
                "hit", "dit", "fram", "frem", "bort", "hjemme", "ute", "inne", "oppe", "nede",
                "borte", "alltid", "aldri", "kanskje"}


def is_function_word(norwegian, data):
    """Служебное слово: ядро по POS (союз/предлог/местоимение/детерминатив), частица «å»,
    или функциональное наречие из белого списка. data — распарсенный dict из word_pool."""
    no = (norwegian or "").strip().lower()
    if no == "å":
        return True
    pos = normalize_pos((data or {}).get("part_of_speech"))   # норв.↔англ. → канон
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


def ramp_kind_of(row):
    """Вид рампы слова (srs.cells.CONTENT/FUNC_*/PHRASE): вся «грязная» классификация
    (парсинг data, словарь служебных, наличие игры у фразы) — здесь, в фасаде;
    чистое ядро srs/ получает готовые булевы."""
    try:
        d = row.get("data")
        d = json.loads(d) if isinstance(d, str) else (d or {})
    except Exception:
        d = {}
    # рампа «фразы» — ТОЛЬКО если есть игровые дистракторы (≥2): легаси-фразы без
    # data.game идут обычной рампой, чтобы не сломать учащиеся слова шагом order
    phrase_playable = (normalize_pos(d.get("part_of_speech")) == "phrase"
                       and len(((d.get("game") or {}).get("distractors")) or []) >= 2)
    return _srs_cells.ramp_kind(
        phrase_playable=phrase_playable,
        function_word=bool(is_function_word(row.get("norwegian"), d)),
        cloze_enabled=CLOZE_ENABLED)


def required_cells(row):
    """Клетки рампы для слова (кортеж-константа из srs.cells)."""
    return _srs_cells.cells_of(ramp_kind_of(row))


def _is_mastered(row, modes):
    """Выучено (делегат srs.status.is_mastered; grandfathering служебных — внутри ядра)."""
    return _srs_status.is_mastered(ramp_kind_of(row), modes)


# Грамм-overlay (тир ★: _grammar_cells/_grammar_element + per-POS хелперы) вынесён в learning_grammar.py;
# реэкспортируется обратно в конце файла (ядро зовёт _grammar_cells/_grammar_element в рантайме).
_LEVEL_ORDER = {lv: i for i, lv in enumerate(LEVELS)}

# --- Зачётный экзамен-ворота (§2.4-A): пейсит приток новых слов ---
# Пороги пачки живут в srs.gates (единый источник, Этап 3) — тут реэкспорт для
# совместимости импортов (exams.py, тесты).
from srs.gates import PACK_FIRST, PACK   # noqa: E402
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


def _next_step(row, modes, audio_on=False):
    """Следующая ступень рампы (делегат srs.steps.next_step): карточка для нового,
    первая несданная клетка, пропуск аудио у контентных при audio ВКЛ, mastered → None."""
    attempts = (row.get("correct") or 0) + (row.get("incorrect") or 0)
    return _srs_steps.next_step(ramp_kind_of(row), modes, attempts=attempts, audio_on=audio_on)


def _review_step(row, modes=None, audio_on=False):
    """Шаг ПОВТОРА (делегат srs.steps.review_step): последняя продуктивная ступень.
    КОНТРАКТ ядра: слово, ждущее слуховой сдачи, на текстовый повтор не выдаётся (None) —
    защита внутри ядра, а не в вызывающих (инцидент #4, 3.07)."""
    return _srs_steps.review_step(ramp_kind_of(row), modes, audio_on=audio_on)


def status_of(row, modes=None):
    """Статус слова из его состояния (делегат srs.status.word_status)."""
    return _srs_status.word_status(
        ramp_kind_of(row), modes,
        attempts=(row.get("correct") or 0) + (row.get("incorrect") or 0),
        strength=row.get("strength") or 0,
        incorrect=row.get("incorrect") or 0,
        reps=row.get("reps") or 0,
        known=bool(row.get("known")),
        archived=bool(row.get("archived")))


def _is_due(row):
    due = row.get("due_at")
    return bool(due) and due <= _now() and not row.get("archived") and not row.get("known")


def _display_status(row, st):
    """Статус для ОТОБРАЖЕНИЯ (счётчики/фильтры/карточки), НЕ для логики сессии:
      • learning + review → 'in_progress' («В процессе»: начато, ещё не выучено);
      • mastered + подошёл срок повтора → 'repeat' («Повторение»): не-серт. с due ИЛИ серт. с audit_due;
      • mastered без срока → 'mastered' («Выучено»);
      • остальное (new/weak/archived) — как есть.
    status_of (внутренняя логика рампы/пулов) при этом не меняется."""
    if st in ("learning", "review"):
        return "in_progress"
    if st == "mastered":
        now = _now()
        is_repeat = ((row.get("due_at") and row["due_at"] <= now and not row.get("certified"))
                     or (row.get("certified") and row.get("audit_due") and row["audit_due"] <= now))
        return "repeat" if is_repeat else "mastered"
    return st


# ---------------- запись результата ответа (SRS) ----------------

_APPLY_LOCK = asyncio.Lock()   # сериализует read-modify-write одного ответа (двойной тап/ретрай/base×grammar)


async def apply_result(user_id: int, pool_id: int, correct: bool, elapsed: float = None,
                       mode: str = None, direction: str = None):
    """Обёртка под локом: иначе два почти-одновременных ответа читают одну старую строку и второй
    commit затирает первый (lost update reps/клетки рампы). Один процесс → asyncio.Lock достаточно."""
    async with _APPLY_LOCK:
        return await _apply_result_inner(user_id, pool_id, correct, elapsed, mode, direction)


async def _apply_result_inner(user_id: int, pool_id: int, correct: bool, elapsed: float = None,
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
        async with db.execute("SELECT norwegian, data, forms FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            wp = await cur.fetchone()
        wrow = {"norwegian": (wp["norwegian"] if wp else None), "data": (wp["data"] if wp else None)}
        cells = required_cells(wrow)
        # грамм-клетки (overlay ★) — записываем результат, но base-рампу/«выучено» они НЕ затрагивают
        gcells = _grammar_cells(wp["norwegian"], wp["data"], wp["forms"]) if wp else []
        modes = {}
        try:
            modes = json.loads(st.get("modes") or "{}")
        except Exception:
            modes = {}
        # окна попыток — нормализуем (старый формат с числами игнорим)
        modes = {k: v for k, v in modes.items() if isinstance(v, str)}
        # клетка задания f"{mode}_{direction}" — только для тестовых типов с заданным направлением
        # (study/без direction — пассив, клетку не трогаем)
        cell = (f"{mode}_{direction}" if (mode and mode != "study" and direction
                                          and direction in _DIR_ALLOWED.get(mode, ())) else None)
        # ── Грамм-overlay (★): фиксируем ТОЛЬКО клетку формы и выходим. base-SRS выученного слова —
        # окно силы (hist), due_at/interval/ease, reps/lapses/streak/correct/incorrect — НЕ трогаем.
        # У overlay нет своего расписания (build_session берёт его по квоте из выученных, без due-проверки),
        # поэтому грамм-ответ не должен двигать base-повтор слова (ни ошибкой откатывать, ни верным
        # раздувать интервал). Дневную активность считаем (реальный ответ → стрик/цель/точность). ──
        if cell and cell in gcells and cell not in cells:
            modes[cell] = "1" if correct else ""
            modes_json = json.dumps(modes, ensure_ascii=False)
            await db.execute(
                "UPDATE user_words SET modes = ?, last_seen = ? WHERE user_id = ? AND pool_id = ?",
                (modes_json, _now(), user_id, pool_id))
            day = _now()[:10]
            await db.execute("""
                INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,1,?)
                ON CONFLICT(user_id, day) DO UPDATE SET answers = answers + 1, correct = correct + ?
            """, (user_id, day, 1 if correct else 0, 1 if correct else 0))
            await db.commit()
            return {"ok": True, "strength": st.get("strength", 0), "due_at": st.get("due_at"),
                    "modes": modes, "mastered": bool(st.get("mastered"))}
        # ── Защита base-SRS: ответ по клетке ФОРМЫ, не попавшей ни в base-рампу, ни в overlay
        # (трек форм со старого бандла без form-флага; регулярная форма вне gcells и т.п.), не должен
        # двигать расписание слова (hist/ease/interval/due). Фиксируем только дневную активность. ──
        if direction and direction not in ("no2int", "int2no") and (not cell or cell not in cells):
            day = _now()[:10]
            await db.execute("""
                INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,1,?)
                ON CONFLICT(user_id, day) DO UPDATE SET answers = answers + 1, correct = correct + ?
            """, (user_id, day, 1 if correct else 0, 1 if correct else 0))
            await db.commit()
            return {"ok": True, "strength": st.get("strength", 0), "due_at": st.get("due_at"),
                    "modes": modes, "mastered": bool(st.get("mastered"))}
        bit = "1" if correct else "0"
        ease = st["ease"]; interval = st["interval_days"]
        modes["hist"] = _push(modes.get("hist", ""), bit, CAPACITY)   # общее окно для силы
        if cell:
            if cell in cells:
                if correct:
                    modes[cell] = "1"              # верно → ступень пройдена
                else:
                    modes[cell] = ""               # ошибка → текущая ступень сброшена
                    # аудио-подтверждение (choice_no2int, слуховая сессия) при ошибке НЕ откатывает
                    # текстовые ступени — слово остаётся «ждёт слух», повторит в след. слуховой партии
                    if cell != _AUDIO_CELL:
                        i = cells.index(cell)
                        if i > 0:                  # ОТКАТ на одну ступень назад (не ниже первой клетки —
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
        mastered_now = 1 if _is_mastered(wrow, modes) else 0   # хранимый флаг: ставим при mastered, снимаем при откате
        await db.execute("""
            INSERT INTO user_words (user_id, pool_id, strength, reps, lapses, ease, interval_days, due_at,
                                    correct, incorrect, streak, archived, modes, last_seen, created_at, mastered)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, pool_id) DO UPDATE SET
                strength=excluded.strength, reps=excluded.reps, lapses=excluded.lapses, ease=excluded.ease,
                interval_days=excluded.interval_days, due_at=excluded.due_at, correct=excluded.correct,
                incorrect=excluded.incorrect, streak=excluded.streak, modes=excluded.modes, last_seen=excluded.last_seen,
                mastered=excluded.mastered
        """, (user_id, pool_id, strength, st["reps"], st["lapses"], ease, interval, due,
              st["correct"], st["incorrect"], st["streak"], st.get("archived", 0), modes_json, _now(), _now(), mastered_now))
        # Замыкание петли забывания (§2.4-B, «вариант A»): забытое на аудите слово было
        # де-сертифицировано (certified=0) и доучивалось в общей очереди. Как только оно снова
        # достигает mastered — СРАЗУ ре-сертифицируем, минуя зачётные ворота, и возвращаем под
        # редкий аудит (audit_due = now + FIRST_AUDIT_DAYS). Признак «ранее сертифицировано/выпало
        # из аудита» — was_certified=1; свежие (никогда не сертифицированные) mastered идут через
        # ворота как раньше (§2.4-A), поэтому условие именно certified=0 AND was_certified=1.
        if mastered_now and not st.get("certified") and st.get("was_certified"):
            await db.execute(
                "UPDATE user_words SET certified = 1, audit_due = ?, audit_interval = ? "
                "WHERE user_id = ? AND pool_id = ?",
                (_due_str(FIRST_AUDIT_DAYS), float(FIRST_AUDIT_DAYS), user_id, pool_id))
        # Цикл «слова↔формы»: свежевыученное ФОРМО-способное слово — в копилку партии
        # (10 шт. → фаза форм, см. build_session/note_cycle_mastered).
        if mastered_now and not (st.get("mastered") or 0) and wp and wp["forms"]:
            try:
                d0 = json.loads(wp["data"]) if wp["data"] else {}
            except Exception:
                d0 = {}
            pos0 = normalize_pos(d0.get("part_of_speech"))
            from .learning_forms import note_cycle_mastered, is_formable
            if is_formable(pos0, wp["forms"], wp["norwegian"]):
                await note_cycle_mastered(db, user_id, pool_id)
        # дневная активность (для стрика/цели/точности/хитмапа)
        day = _now()[:10]
        await db.execute("""
            INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,1,?)
            ON CONFLICT(user_id, day) DO UPDATE SET answers = answers + 1, correct = correct + ?
        """, (user_id, day, 1 if correct else 0, 1 if correct else 0))
        await db.commit()
        return {"ok": True, "strength": strength, "due_at": due, "modes": modes, "mastered": bool(mastered_now)}
    finally:
        await _release(db)


async def set_status(user_id: int, pool_id: int, action: str):
    """action: know (в архив) | known (в корзину «Знаю») | reset (сброс) | unarchive (вернуть в ротацию)."""
    db = await _conn()
    try:
        if action == "know":
            m = {c: "1" for c in ALL_CELLS}   # все клетки рампы пройдены (и choice/build/input, и cloze)
            m["hist"] = "1" * CAPACITY
            fields = "known=0, archived=1, strength=100, reps=MAX(reps,3), modes=?, due_at=?, mastered=1"
            args = (json.dumps(m, ensure_ascii=False), _due_str(120))
        elif action == "known":
            # «Уже знаю»: знакомое слово → корзина «Знаю» (known=1), ВНЕ ротации. Это НЕ «Выучено»:
            # mastered НЕ ставим и прогресс уровня НЕ двигаем (см. status_of → 'known'). Срок повтора
            # убираем (не всплывёт). Историю попыток не трогаем — слово просто «отложено как знакомое».
            fields = "known=1, mastered=0, archived=0, due_at=NULL"
            args = ()
        elif action == "reset":
            fields = ("known=0, archived=0, strength=0, reps=0, lapses=0, ease=2.5, interval_days=0, "
                      "correct=0, incorrect=0, streak=0, modes=NULL, due_at=NULL, mastered=0, "
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

async def _fetch_user_words(db, user_id, set_id=None):
    """Все слова пользователя (уникальные по пулу) + состояние SRS + темы.
    set_id задан → ограничиваемся словами ОДНОГО набора (для дрилла «Учить набор»):
    берём все слова набора независимо от флага studying (явная тренировка по набору)."""
    if set_id is not None:
        # слова конкретного набора пользователя (проверка владения по d.user_id)
        src_sql = """
              SELECT dw.pool_id FROM dict_words dw
              JOIN dictionaries d ON d.id = dw.dict_id
              WHERE d.user_id = ? AND d.id = ?
        """
        src_params = (user_id, set_id)
    else:
        src_sql = """
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
        """
        src_params = (user_id, user_id, user_id)
    async with db.execute(f"""
        SELECT wp.id AS pool_id, wp.norwegian, wp.data, wp.level, wp.freq, wp.forms, {has_tts_expr("wp")} AS has_tts,
               uw.strength, uw.reps, uw.lapses, uw.ease, uw.interval_days, uw.due_at,
               uw.correct, uw.incorrect, uw.streak, uw.archived, uw.modes, uw.last_seen,
               uw.certified, uw.audit_due, uw.audit_interval, uw.was_certified, uw.mastered, uw.known
        FROM ({src_sql}) up
        JOIN word_pool wp ON wp.id = up.pool_id
        LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
    """, (*src_params, user_id)) as cur:
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
    st = status_of(r, modes)
    _rcells = required_cells(r)   # ступени рампы слова (контент: 4 REQUIRED_CELLS; служебное: 3 cloze)
    return {
        "pool_id": r["pool_id"], "no": r["norwegian"],
        "translate": data.get("translate", {}), "part_of_speech": data.get("part_of_speech", ""),
        "level": r.get("level"), "freq": r.get("freq"), "topics": r.get("topics", []), "hasTts": bool(r.get("has_tts")),
        # status — ВНУТРЕННИЙ (рампа/уровни/пулы); dstatus — для отображения (см. _display_status)
        "status": st, "dstatus": _display_status(r, st), "strength": r.get("strength") or 0, "due_at": r.get("due_at"),
        # прогресс по рампе (для «в процессе»): сколько ступеней пройдено из total
        "ramp": {"done": sum(1 for c in _rcells if modes.get(c) == "1"), "total": len(_rcells)},
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
        # фильтр по ОТОБРАЖАЕМОМУ статусу (new/in_progress/repeat/weak/mastered/archived).
        # 'repeat' = выученное, подошёл срок повтора (см. _display_status) — то же, что чип «повторить».
        items = [w for w in items if w["dstatus"] == status]
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
    # эффективный срок «когда повторять»: у сертифицированных — аудит (audit_due), иначе due_at.
    # «Скоро повторять» = про повтор ВЫУЧЕННОГО. Поэтому новые (нет расписания), архивные (отложены)
    # и «в процессе» (ещё не выучено, активное изучение) — в КОНЕЦ ("9999"), чтобы не подмешивались
    # к предстоящим повторам. Сверху — выученное по эффективному сроку (как счётчик на карточке).
    _next_at = lambda w: ("9999" if w.get("dstatus") in ("new", "archived", "in_progress")
                          else ((w["audit_due"] if (w.get("certified") and w.get("audit_due")) else w.get("due_at")) or "9999"))
    _keys = {
        "alpha":    _no,
        "due":      lambda w: (_next_at(w), _no(w)),
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
    due = sorted([w for w in items if w["due"] and not w["certified"]], key=lambda w: w["due_at"] or "")
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

from srs.gates import WIP_LIMIT   # noqa: E402 — лимит «в работе»: единый источник srs.gates
NEW_PER_SESSION = 6   # потолок НОВЫХ слов (карточек-знакомств) за одну сессию — остальное добиваем
                      # заданиями рампы. Состав растёт сам: 6 карт → 6+6 заданий → 6+12 → 2+18 (до size).
PHRASE_BUFFER = 2     # сколько НЕначатых устойчивых выражений держим доступными в Учёбе (тонкий ручеёк):
                      # подмешиваются новыми карточками наравне со словами, делят cap_new (NEW_PER_SESSION).
PHRASE_PREREQ_CAP = 3 # макс. невыученных дистракторов фразы досыпаем как prereq-лексику за сессию
                      # (когда у order-игры мало узнаваемых дистракторов — учим их, фраза ждёт).


async def _attach_choice_options(session, lang, n=3):
    """Для choice-элементов сессии дотягиваем варианты ответа (как /pool/{id}/distractors),
    чтобы фронт получил ВСЁ нужное одним запросом. IO-обвязка Этапа 7: соседи —
    из РЕЗИДЕНТНОГО кеша эмбеддингов (embcache, один matvec на все слова + один
    батч-запрос за соседями, без sqlite-KNN), сам расчёт — чистый
    session/distractors.options_patch; применяется только контентным choice
    (свои варианты грамм-тира патч не перезапишет by construction)."""
    items = _distractors.choice_targets(session)
    if not items:
        return
    import embcache
    from db import get_pool_words_by_ids
    pids = [e["pool_id"] for e in items]
    nbr_map = await embcache.candidates_for(pids, 45)
    all_nb = {i for pid in pids for i in (nbr_map.get(pid) or [])}
    words = await get_pool_words_by_ids(list(all_nb)) if all_nb else {}
    patch = _distractors.options_patch(items, neighbors=nbr_map, words=words, lang=lang, n=n)
    if not any(p["options"] for p in patch.values()):
        # деградация (холодный embcache после рестарта и т.п.) — говорим в лог, не молчим:
        # у всех choice-элементов пустые варианты = фронт покажет вопрос без ответов
        logger.warning("options_patch: пустые варианты у всех %d choice-элементов "
                       "(embcache холодный?)", len(items))
    _distractors.apply_patch(session, patch)


async def build_session(user_id, size=20, lang="ru", set_id=None):
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
    set_id задан → ДРИЛЛ ПО НАБОРУ: берём только слова этого набора, без авто-добора из
    «Базы» и без лимита WIP/ворот на ввод новых (явная тренировка — даём учить что выбрано).
    Возвращает [{pool_id, no, translate, mode, direction, step}], не больше size."""
    scoped = set_id is not None
    _t0 = time.monotonic(); _tm = {}   # [perf] тайминг фаз сборки сессии
    from .users import get_user_new_per_session, get_user_grammar, get_user_grammar_pos, get_user_audio  # ленивый импорт
    cap_new = await get_user_new_per_session(user_id, NEW_PER_SESSION)   # порция новых за сессию (настройка профиля)
    audio_on, _listen_pack = await get_user_audio(user_id)   # аудио ВКЛ → choice_no2int откладываем в слуховую сессию
    async def _load():
        """Прочитать слова пользователя + флаг тормоза, разложить по статусам."""
        db = await _conn()
        try:
            rows = await _fetch_user_words(db, user_id, set_id)
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
        # бюджеты/ворота — чистые именованные правила srs.gates (Этап 3): слуховые
        # слова слот «в работе» не занимают; критерий пачки/тормоза — в одном месте
        in_work = sum(1 for e in enriched if _gates.counts_toward_wip(
            e["status"], ramp_kind_of(e["row"]), e["modes"], audio_on=audio_on))
        pack_n = sum(1 for e in enriched if e["status"] == "mastered" and not _is_certified(e["row"]))
        had_cert = any(_is_certified(e["row"]) for e in enriched)
        gate_open = _gates.exam_gate_open(pack_n, had_cert=had_cert, throttled=throttled)
        return enriched, in_work, gate_open

    enriched, in_work, gate_open = await _load()
    _tm["load"] = time.monotonic() - _t0

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
    if not scoped and not gate_open:
        # залоченные пословным порогом новые служебные не считаем «доступными новыми» — иначе они
        # держат new_avail>0 и блокируют добор контентных, которыми сами же и разблокируются.
        # Фразы тоже исключаем — у них свой ручеёк (suggest_phrases ниже), иначе они держали бы
        # new_avail>0 и блокировали добор слов, когда свои слова у юзера кончились.
        new_avail = sum(1 for e in enriched if e["status"] == "new" and not _func_locked(e)
                        and (e["data"].get("part_of_speech") != "phrase"))
        if new_avail == 0 and in_work < WIP_LIMIT:
            _ts = time.monotonic()
            level = await estimate_level(user_id)
            res = await suggest_words(user_id, count=WIP_LIMIT - in_work, level=level, allow_func=func_gate_ok)
            if res.get("added"):
                # перечитываем слова после добора и продолжаем обычную сборку
                enriched, in_work, gate_open = await _load()
                content_known = _content_known(enriched)
                func_gate_ok = content_known >= FUNC_GATE
            _tm["suggest"] = time.monotonic() - _ts

    # УСТОЙЧИВЫЕ ВЫРАЖЕНИЯ: тонкий ручеёк НЕЗАВИСИМО от слов — держим PHRASE_BUFFER неначатых фраз
    # доступными (кумулятивно по уровню). Подмешиваются новыми карточками наравне со словами, делят
    # cap_new. ВАЖНО: после авто-добора слов выше (иначе фразы в new_avail подавили бы тот добор).
    if not scoped and not gate_open:
        _tp = time.monotonic()
        dbp = await _conn()
        try:
            supply = await _phrase_supply(dbp, user_id)
        finally:
            await _release(dbp)
        if supply < PHRASE_BUFFER:
            pres = await suggest_phrases(user_id, count=PHRASE_BUFFER - supply)
            if pres.get("added"):
                enriched, in_work, gate_open = await _load()
                content_known = _content_known(enriched)
                func_gate_ok = content_known >= FUNC_GATE
        _tm["phrases"] = time.monotonic() - _tp

    # пулы/отбор/кулдаун — чистые шаги session.pools (Этап 4); БД-специфика (предикаты,
    # степы ядра, настройки юзера) инжектится замыканиями
    def attempts(e):
        return _pools.attempts_of(e)

    pools = _pools.build_pools(enriched, is_certified=_is_certified,
                               is_function_word=is_function_word)
    cand = _pools.select_candidates(
        pools,
        next_step=lambda e: _next_step(e["row"], e["modes"], audio_on),
        review_step=lambda e: _review_step(e["row"], e["modes"], audio_on=audio_on),
        func_locked=_func_locked)
    cooldown_cut = (datetime.utcnow() - timedelta(minutes=COOLDOWN_MIN)).isoformat()
    ordered = _pools.apply_cooldown(cand, cooldown_cut=cooldown_cut)

    # ── ДОСРОЧНЫЕ ПОВТОРЫ (анти-тупик ветерана, 3.07): повторов due нет, новые заперты
    # WIP-лимитом/воротами, а формы могут быть выключены тумблером — раньше сессия
    # приходила ПУСТОЙ («В процессе 23, Повторение 0» → тупик до завтрашних due).
    # Даём начатые слова с БЛИЖАЙШИМ due на повтор раньше срока: SRS не страдает —
    # интервал пересчитается от сегодняшнего ответа. ──
    early_review = False
    if not ordered and not scoped:
        ordered = _pools.early_review_pool(
            enriched, size=size,
            review_step=lambda e: _review_step(e["row"], e["modes"], audio_on=audio_on))
        early_review = bool(ordered)

    # ── Дистракторы order-игры (устойчивые выражения): показываем только УЗНАВАЕМЫЕ — уровень ≤
    # уровня юзера ИЛИ уже выученные. Уровни дистракторов берём из Базы (они туда заведены). Если у
    # фразы наберётся <2 годных дистракторов — фразу пока пропускаем, а её невыученные (но pool-backed)
    # дистракторы досыпаем как prereq-лексику (после цикла), чтобы фраза разблокировалась позже. ──
    order_cands = [c for c in ordered if c[1][1] == "order"]
    distr_level, distr_pid, known_no, prereq = {}, {}, set(), []
    user_rank = None
    if order_cands:
        ulvl = await estimate_level(user_id)
        user_rank = _LEVEL_ORDER.get(ulvl, 0)
        known_no = {(e["row"]["norwegian"] or "").lower() for e in enriched
                    if e["status"] == "mastered" or e["row"].get("known")}
        words = {(d or "").strip().lower()
                 for e, _ in order_cands for d in ((e["data"].get("game") or {}).get("distractors") or [])}
        words.discard("")
        if words:
            dbl = await _conn()
            try:
                marks = ",".join("?" for _ in words)
                async with dbl.execute(
                    f"SELECT id, LOWER(norwegian) AS no, level FROM word_pool WHERE LOWER(norwegian) IN ({marks})",
                    list(words)) as cur:
                    for r in await cur.fetchall():
                        distr_pid[r["no"]] = r["id"]
                        if r["level"] in _LEVEL_ORDER:
                            distr_level[r["no"]] = _LEVEL_ORDER[r["level"]]
            finally:
                await _release(dbl)

    def _distr_ok(d):
        dl = (d or "").strip().lower()
        if dl in known_no:                       # уже выучено юзером — узнаваемо
            return True
        r = distr_level.get(dl)
        return r is not None and user_rank is not None and r <= user_rank   # уровень ≤ юзера

    # cloze-кэш для служебных слов на стадии cloze (предложения готовим заранее, фоном)
    cloze_pids = [e["row"]["pool_id"] for (e, step) in ordered if step[1] == "cloze"]
    cloze_map = {}
    if cloze_pids:
        _tc = time.monotonic()
        dbc = await _conn()
        try:
            cloze_map = await get_cloze_map(dbc, user_id, cloze_pids)
        finally:
            await _release(dbc)
        _tm["cloze"] = time.monotonic() - _tc

    # ── Грамм-тир ★ и ЦИКЛ «слова ↔ формы» (form_cycle). Фаза words: сессия учит слова, из форм —
    # ТОЛЬКО подошедшие повторы (due) + pronoun-overlay; выученные слова копятся в партию (см.
    # apply_result → note_cycle_mastered). Фаза forms: сессия дрилит формы партии (FORMS_SESSION_SHARE
    # слотов, ≤2 клетки на слово, новых СЛОВ не вводим), остаток — due/прогресс слов; партия сдана
    # (все клетки produce+interval≥1) → флип обратно в words. Формы на «выучено»/CEFR не влияют. ──
    grammar_on = await get_user_grammar(user_id)
    grammar_pos = await get_user_grammar_pos(user_id) if grammar_on else {}   # пер-POS тумблеры
    fplan = _forms_phase.empty_plan(cap_new)   # грамматика выкл / дрилл → фазы нет, cap_new НЕТРОНУТ
    if grammar_on and not scoped:
        from .learning_forms import (form_cells_for, load_form_states, get_form_cycle, save_form_cycle,
                                     FORM_CYCLE_BATCH, FORMS_SESSION_SHARE)

        def _group_of(e):   # POS → группа грамм-тира с учётом пер-POS тумблеров настроек
            g = _GRAMMAR_GROUP.get(normalize_pos((e["data"] or {}).get("part_of_speech")))
            return g if g and grammar_pos.get(g, True) else None

        def _cells_for(e, fdict):
            return form_cells_for(normalize_pos((e["data"] or {}).get("part_of_speech")),
                                  fdict, e["row"]["norwegian"])

        def _overlay_pending(e, fdict):
            return [c for c in _grammar_cells(e["row"]["norwegian"], e["data"], fdict)
                    if e["modes"].get(c) != "1"]

        # слова, уже взятые в base-сессию (в т.ч. due-повторы выученных), грамматикой НЕ
        # резервируем — отсеет build_universe (иначе слот квоты сгорел бы на дедупе ниже)
        cands, finfo = _forms_phase.build_universe(
            enriched, ordered_pids={e["row"]["pool_id"] for e, _ in ordered},
            group_of=_group_of, pronoun_forms=PRONOUN_PARADIGM.get, cells_for=_cells_for)
        fstates = await load_form_states(user_id, list(finfo)) if finfo else {}
        blocked_new = _gates.new_words_blocked(in_work, gate_open, wip_limit=WIP_LIMIT)
        fplan = _forms_phase.plan_forms_phase(
            cands, finfo, fstates=fstates, cycle_state=await get_form_cycle(user_id),
            now_s=_now(), cap_new=cap_new, size=size,
            base_servable=sum(1 for e2, _s2 in ordered if attempts(e2) > 0 or not blocked_new),
            batch_size=FORM_CYCLE_BATCH, session_share=FORMS_SESSION_SHARE,
            overlay_pending=_overlay_pending)
        if fplan["save_cycle"]:
            await save_form_cycle(user_id, *fplan["save_cycle"])
    grammar_picks = fplan["picks"]   # [(kind 'form'|'overlay', e, cell, fdict, stage|None)]
    cycle_phase, cycle_left, cycle_cells = fplan["phase"], fplan["cycle_left"], fplan["cycle_cells"]
    cap_new = fplan["cap_new"]   # фаза forms новых слов не вводит: 0 приходит ПОЛЕМ плана, не мутацией

    # ФАЗА СЛОВ: грамматики НЕТ ВООБЩЕ (решение юзера — «сессия слов = только слова +
    # карточки новых в конце»); формы и местоим-overlay живут в фазе форм — цикл короткий,
    # интервалы почти не едут. Отбор заданий фазы форм — session/forms_phase.plan_forms_phase.
    base_budget = max(0, size - len(grammar_picks))   # под контент — остаток после грамм-квоты

    session = []
    new_added = 0                                            # сколько новых карточек уже взяли в эту сессию
    comp = {"fresh": 0, "review": 0, "weak": 0, "progress": 0, "phrases": 0, "grammar": 0}   # состав (для честной кнопки старта); phrases — выражения, grammar — грамм-overlay
    for e, step in ordered:
        # Фаза ФОРМ: хвост сессии — ТОЛЬКО повторы выученных (приходят вводом). Слабые/начатые
        # слова середины рампы ждут фазы слов — иначе в «сессию форм» вклинивается выбор перевода
        # недоученного слова и ломает ощущение режима. Повторы не морозим (интервалы святы).
        if cycle_phase == "forms" and e["row"].get("mastered") != 1:
            continue
        is_new = attempts(e) == 0
        # Порционное знакомство: потолок НОВЫХ карточек за сессию (NEW_PER_SESSION) действует ВСЕГДА,
        # в т.ч. в дрилле по набору (scoped) — иначе все новые слова набора валятся карточками сразу.
        # Лимит WIP / ворота экзамена — только вне дрилла (явная тренировка их не ограничивает).
        if is_new and (new_added >= cap_new or (not scoped and (in_work >= WIP_LIMIT or gate_open))):
            continue
        cell, mode, direction = step
        pid = e["row"]["pool_id"]
        data = e["data"] or {}   # уже распарсено в _load — не парсим data повторно на каждый элемент
        el = _make_element(
            pool_id=pid, no=e["row"]["norwegian"],
            translate=data.get("translate", {}),
            part_of_speech=data.get("part_of_speech", ""),
            gloss=data.get("gloss"), example=data.get("example"),  # для карточки служебного (Ф2)
            forms=(json.loads(e["row"]["forms"]) if e["row"].get("forms") else None),  # колонка wp.forms (не data!) — для артикля (en/ei/et) сущ. и «å» глаг.
            mode=mode, direction=direction, step=cell,
            # повтор = ХРАНИМЫЙ флаг mastered (слово было доведено до выученного и теперь на повторении),
            # а не эвристика. Слова в первом прохождении рампы повтором НЕ считаются.
            repeat=(e["row"].get("mastered") == 1),
        )
        if mode == "order":
            # игра «порядок слов»: показываем только УЗНАВАЕМЫЕ дистракторы (уровень ≤ юзера / выучено)
            raw = (data.get("game") or {}).get("distractors") or []
            ok = [d for d in raw if _distr_ok(d)]
            if len(ok) < 2:
                # мало узнаваемых дистракторов → фразу пока не показываем; её невыученные (но
                # pool-backed) дистракторы — в prereq-лексику, чтобы разблокировать фразу позже
                prereq.extend(dl for d in raw
                              if not _distr_ok(d) and (dl := (d or "").strip().lower()) in distr_pid)
                continue
            el["distractors"] = ok[:3]
        if mode == "cloze":
            items = cloze_map.get(pid)
            idx = (int(direction) - 1) if str(direction).isdigit() else 0
            if not items or idx >= len(items):
                # cloze ещё не сгенерён — запускаем фоном, это слово сейчас пропускаем (придёт позже)
                asyncio.create_task(generate_cloze(user_id, pid))
                continue
            el["cloze"] = items[idx]
        session.append(el)
        if data.get("part_of_speech") == "phrase":
            comp["phrases"] += 1   # устойчивое выражение в сессии (доп. счётчик поверх fresh/progress)
        if is_new:
            in_work += 1       # ввели новое — слот занят
            new_added += 1
            comp["fresh"] += 1
        elif e["row"].get("mastered") == 1:
            comp["review"] += 1   # выученное на повторении
        elif e["status"] == "weak":
            comp["weak"] += 1
        else:
            comp["progress"] += 1  # начатое, ещё не выученное
        if len(session) >= base_budget:   # base-бюджет = size − грамм-квота (overlay добьём ниже)
            break
    # prereq-лексика: невыученные дистракторы пропущенных фраз → в скрытый авто-словарь, чтобы фразы
    # разблокировались (дистракторы станут узнаваемыми). Капим и дедупим; уважаем ворота/дрилл.
    if prereq and not scoped and not gate_open:
        from .dictionaries import add_word_to_dict, get_or_create_hidden_dict
        hid = await get_or_create_hidden_dict(user_id)
        for d in list(dict.fromkeys(prereq))[:PHRASE_PREREQ_CAP]:
            pid_pre = distr_pid.get(d)
            if pid_pre:
                await add_word_to_dict(user_id, hid, pid_pre)

    # грамм-тир ★: трек форм (сущ./глаг./прил.) + pronoun-overlay — добиваем выученными словами;
    # не дублируем слово, если оно уже попало в сессию обычным упражнением (повтор и т.п.)
    if grammar_picks:
        from .learning_forms import form_element   # ленивый импорт (симметрично load_form_states выше)
    in_session = {el["pool_id"] for el in session}
    for kind, e, cell, fdict, stage in grammar_picks:
        if e["row"]["pool_id"] in in_session:
            continue
        el = (form_element(e["row"], fdict, e["data"], cell, stage) if kind == "form"
              else _grammar_element(e["row"], cell, fdict, e["data"]))
        if el:
            session.append(el)
            in_session.add(e["row"]["pool_id"])
            comp["grammar"] += 1
    # фаза цикла «слова↔формы» — для честной кнопки старта (чип «сессия форм» + остаток партии)
    comp["phase"] = cycle_phase
    if cycle_phase == "forms":
        comp["formsLeft"] = cycle_left           # слов партии с несданными формами
        comp["formsCellsLeft"] = cycle_cells     # несданных ФОРМ (клеток) до возврата новых слов
        # сессия ФОРМ должна и НАЧИНАТЬСЯ с форм: до сортировки формы висели в хвосте (эмиссия
        # после base-цикла) — юзер открывал «сессию форм», а первым шёл перевод слова. sort
        # стабилен: внутри форм порядок партии сохранён, base-повторы уходят в хвост.
        session.sort(key=lambda el: not el.get("form_track"))

    # карточки-знакомства (новые слова, step == "card") — в КОНЕЦ сессии: сначала упражнения по уже
    # начатым словам, потом интро новых. sort стабилен → относительный порядок внутри групп сохранён.
    session.sort(key=lambda el: el.get("step") == "card")
    _ta = time.monotonic()
    await _attach_choice_options(session, lang)
    _tm["choice"] = time.monotonic() - _ta
    comp["total"] = len(session)
    # Диагностика comp.reason — единая лестница session.reason.session_reason (и «сессия из
    # досрочных повторов» и ПУСТАЯ сессия ветерана: почему пусто — повторы не due; новые заперты
    # пачкой/WIP; слух ждёт). has_audio_pending считаем лениво — только у пустой сессии.
    reason = _reason.session_reason(
        session_len=len(session), scoped=scoped, early_review=early_review,
        has_audio_pending=(not session and audio_on and any(
            _is_audio_pending(e["row"], e["modes"], e["data"]) for e in enriched)),
        gate_open=gate_open, in_work=in_work, wip_limit=WIP_LIMIT)
    if reason:
        comp["reason"] = reason
    _tm["total"] = time.monotonic() - _t0
    if _tm["total"] > 1.0:   # канарейка: логируем ТОЛЬКО медленную сборку (>1с) с разбивкой фаз
        logger.warning("slow build_session uid=%s n=%d %s", user_id, len(session),
                       " ".join(f"{k}={v * 1000:.0f}ms" for k, v in _tm.items()))
    return {"words": session, "composition": comp}


# ---------------- слуховая сессия (аудио-подтверждение выученного текстом) ----------------
def _is_audio_pending(row, modes, data):
    """«Ждёт слух» (делегат srs.status.is_audio_pending): текстовая рампа сдана,
    аудио-клетка нет; только контентные слова."""
    kind = ramp_kind_of({"norwegian": row.get("norwegian"), "data": data})
    return _srs_status.is_audio_pending(kind, modes)


async def _audio_pending_rows(user_id):
    """[(row, data, modes)] слов, ждущих слух (см. _is_audio_pending)."""
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
    finally:
        await _release(db)
    out = []
    for r in rows:
        try:
            modes = json.loads(r.get("modes") or "{}")
        except Exception:
            modes = {}
        try:
            data = json.loads(r["data"]) if r.get("data") else {}
        except Exception:
            data = {}
        if _is_audio_pending(r, modes, data):
            out.append((r, data, modes))
    return out


async def listen_status(user_id):
    """Сколько слов «ждёт слух» + порог партии + готова ли слуховая сессия (pending >= порог).
    Аудиозадания выключены → audio False (откладывать нечего, choice_no2int идёт в дневной рампе)."""
    from .users import get_user_audio
    audio_on, pack = await get_user_audio(user_id)
    if not audio_on:
        return {"pending": 0, "pack": pack, "ready": False, "audio": False}
    rows = await _audio_pending_rows(user_id)
    return {"pending": len(rows), "pack": pack, "ready": len(rows) >= pack, "audio": True}


async def build_listen_session(user_id, size=20, lang="ru"):
    """Слуховая партия: аудио-подтверждение (choice_no2int) слов, прошедших текстовую рампу. Сдача →
    слово полностью выучено (apply_result закрывает 4-ю клетку). Ошибка слух НЕ откатывает текстовые
    ступени. Только при включённых аудиозаданиях. Дольше всех ждущие — первыми. {words, composition}."""
    from .users import get_user_audio
    audio_on, _pack = await get_user_audio(user_id)
    if not audio_on:
        return {"words": [], "composition": {"listen": 0, "total": 0}}
    pending = await _audio_pending_rows(user_id)
    pending.sort(key=lambda rdm: rdm[0].get("last_seen") or "")   # дольше всех не виделись — первыми
    session = []
    for r, data, _modes in pending[:size]:
        session.append(_make_element(
            pool_id=r["pool_id"], no=r["norwegian"],
            translate=data.get("translate", {}),
            part_of_speech=data.get("part_of_speech", ""),
            gloss=data.get("gloss"), example=data.get("example"),  # union Этапа 6 (было только в дневной)
            forms=(json.loads(r["forms"]) if r.get("forms") else None),
            mode="choice", direction="no2int", step=_AUDIO_CELL,
            listen=True,   # фронт: проигрывать аудио, текст скрыт (слуховое узнавание)
            repeat=(r.get("mastered") == 1),
        ))
    await _attach_choice_options(session, lang)
    return {"words": session, "composition": {"listen": len(session), "total": len(session)}}


# ---------------- cloze для служебных слов (A1) — вынесено в db/learning_cloze.py ----------------
# get_cloze_map / generate_cloze / _blank_example реэкспортируются в конце файла
# (ядро зовёт get_cloze_map/generate_cloze в рантайме; _blank_example нужен ещё exams.py).


# Зачётный экзамен-ворота и аудит забывания вынесены в db/exams.py (реэкспорт в конце файла).
# Подсказки/статистика (learning_stats, estimate_level, suggest_words, suggest_phrases,
# _phrase_supply, next_new_cards, _own_new_card_rows) вынесены в db/learning_suggest.py;
# реэкспорт в конце файла ПОСЛЕДНИМ (зависят от exams/placement). Ядро (build_session) зовёт
# estimate_level/_phrase_supply/suggest_phrases в рантайме.


# --- Реэкспорт вынесенных модулей (чтобы `from db.learning import …` и db/__init__ не менялись) ---
from .learning_grammar import _grammar_cells, _grammar_element  # noqa: E402,F401  (ядро зовёт в рантайме)
from .learning_cloze import get_cloze_map, generate_cloze, _blank_example  # noqa: E402,F401  (_blank_example нужен exams)
from .leaderboard import get_activity, learning_leaderboard  # noqa: E402,F401
from .placement import (  # noqa: E402,F401
    get_start_level, set_start_level, build_placement, grade_placement, seed_starter,
    PLACEMENT_PASS, PLACEMENT_MIN, STARTER_GOAL,
)
from .exams import (  # noqa: E402,F401
    gate_status, new_words_blocked, build_gate_exam, grade_gate_exam,
    build_audit, grade_audit, audit_throttled, _pack_rows,
    _is_certified, _audit_throttled,   # нужны ядру (build_session/get_due/learning_stats) — резолвятся в рантайме
)
# ПОСЛЕДНИМ — зависит от exams/placement (импортит из них на верхнем уровне):
from .learning_suggest import (  # noqa: E402,F401
    learning_stats, estimate_level, suggest_words, suggest_phrases, _phrase_supply, next_new_cards,
)
