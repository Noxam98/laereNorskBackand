"""Трек ФОРМ (bokmål): рампа изучения словоформ поверх выученного слова — ЧИСТАЯ логика (без БД).

Отличие от грамм-overlay (learning_grammar): трек гоняет КАЖДУЮ форму слова (в т.ч. регулярную),
имеет собственную рампу карточка→выбор→ввод и СВОЁ расписание (SM-2 на клетку — schedule_form;
хранилище form_srs и apply подключаются на слое БД отдельным шагом). Дистракторы «выбора» —
динамическая подмена окончаний из morphology.*_form_options: различаем нужную форму среди
правдоподобных ошибок (gå→*gådde, grønn→*grønnt), не «угадайка» по чужим словам.

Хранилище — form_srs (user_id, pool_id, cell → stage/ease/interval/due): apply_form_result,
load_form_states. Чистая часть (form_element/schedule_form) — ниже, вызывается и сессией, и apply.

Цикл «СЛОВА ↔ ФОРМЫ» (form_cycle): фазы чередуются по кругу — в фазе words сессии учат слова
(из форм только подошедшие повторы), выученные формо-способные слова копятся в партию (batch);
набралось FORM_CYCLE_BATCH → фаза forms: сессии дрилят формы партии (новых слов не вводим),
пока ВСЕ клетки партии не сданы (produce → interval≥1) — тогда снова words. Повторы обоих
треков (due) живут в любой фазе — SRS не замораживается."""
import json
import os
from datetime import datetime, timedelta

from pos import normalize_pos
from .core import _conn, _release, _now
from morphology import (
    strip_aux,
    noun_form_options, verb_form_options, adj_form_options,
    NOUN_FORM_CELLS, VERB_FORM_CELLS, ADJ_FORM_CELLS, UNCOUNTABLE_NOUNS,
)

# Рампа одной формы: показать карточку → выбрать верную среди подменённых окончаний → набрать самому.
FORM_STAGES = ("card", "choose", "produce")

# Части речи с морфологией форм (местоим./детерминативы — курируемая парадигма, не сюда).
FORM_CELLS_BY_POS = {
    "noun": NOUN_FORM_CELLS,
    "verb": VERB_FORM_CELLS,
    "adjective": ADJ_FORM_CELLS,
}

# Клетка формы → (поле в forms, метка-подпись для фронта i18n). gender — особая (выбор артикля).
_FORM_FIELD = {
    "gender":      ("gender",      "gender"),
    "indef_pl":    ("indef_pl",    "indef_pl"),
    "def_sg":      ("def_sg",      "def_sg"),
    "def_pl":      ("def_pl",      "def_pl"),
    "present":     ("present",     "present"),
    "past":        ("past",        "past"),
    "perfect":     ("perfect",     "perfect"),
    "neuter":      ("neuter",      "neuter"),
    "plural":      ("plural",      "plural_adj"),
    "comparative": ("comparative", "comparative"),
    "superlative": ("superlative", "superlative"),
}

# SM-2 (как база, см. learning.apply_result) — держим тут, чтобы модуль был самодостаточным.
_EASE_START, _EASE_MAX, _EASE_MIN = 2.5, 3.0, 1.3
_INTERVAL_CAP = 365
_FAST_SEC = 6.0


def _pos_of(data):
    try:
        d = json.loads(data) if isinstance(data, str) else (data or {})
    except Exception:
        d = {}
    return normalize_pos(d.get("part_of_speech"))


def parse_forms(forms):
    try:
        return json.loads(forms) if isinstance(forms, str) else (forms or {})
    except Exception:
        return {}


# Маркеры «формы не существует» из LLM-заполнения (n/a у несравнимых прилагательных и т.п.):
# такие клетки НЕ дрилим (иначе просили бы ввести «n/a») и на фронте показываем «нет формы».
_JUNK_FORMS = {"n/a", "na", "-", "–", "—", "none", "null", "ingen"}


def cell_value(pos, forms, cell):
    """Значение клетки формы ('' если формы нет/мусор-маркер). Перфект глагола дрилим как
    ПРИЧАСТИЕ (без 'har' — вспом. постоянен)."""
    if cell == "gender":
        val = (forms.get("gender") or "").strip()
    else:
        field = _FORM_FIELD.get(cell, (cell, cell))[0]
        val = (forms.get(field) or "").strip()
        if pos == "verb" and cell == "perfect" and val:
            val = strip_aux(val)
    return "" if val.lower() in _JUNK_FORMS else val


def form_cells_for(pos, forms, no=None):
    """Клетки форм, реально доступные слову: часть речи из FORM_CELLS_BY_POS + форма присутствует и
    не пуста (несклоняемое/отсутствующее — пропускаем; перифразы 'mer/mest …' — тоже, это не одна форма).
    Неисчисляемые сущ. (bruk, vann…): мн.ч. в речи не употребляется — клетки мн.ч. не дрилим."""
    cells = FORM_CELLS_BY_POS.get(pos)
    if not cells:
        return []
    mass = pos == "noun" and (forms.get("uncountable") is True
                              or (no and str(no).strip().lower() in UNCOUNTABLE_NOUNS))
    out = []
    for c in cells:
        if mass and c in ("indef_pl", "def_pl"):
            continue
        v = cell_value(pos, forms, c)
        if v and " " not in v:          # 'mer praktisk' и т.п. — не одна словоформа, мимо
            out.append(c)
    return out


def is_formable(pos, forms, no=None):
    """ЕДИНЫЙ предикат «слово формо-способно» (Этап 5): трек форм ведём только для
    noun/verb/adjective и только если есть реальные клетки (form_cells_for непуст).
    Им пользуются ОБА края цикла: apply_result (копилка партии note_cycle_mastered)
    и build_session (вселенная трека) — расхождение критериев = слово копится в
    партию, но сессия его не дрилит (или наоборот)."""
    return pos in ("noun", "verb", "adjective") and bool(
        form_cells_for(pos, forms if isinstance(forms, dict) else parse_forms(forms), no))


def form_options(pos, no, forms, cell, n=3):
    """(correct, [distractors]) для клетки формы — диспетч в морфологию по части речи."""
    if pos == "noun":
        return noun_form_options(no, (forms.get("gender") or "").strip(), forms, cell, n)
    if pos == "verb":
        return verb_form_options(no, forms, cell, n)
    if pos == "adjective":
        return adj_form_options(no, forms, cell, n)
    return (None, [])


def form_element(row, forms, data, cell, stage):
    """Сессионный элемент трека форм для клетки+ступени. Совместим с контрактом грамм-элемента
    (mode/direction/target/prompt) + флаг form_track:True (фронт роутит ответ в трек, не в base-рампу).
    card — пассивный показ формы; choose — выбор среди подменённых окончаний; produce — ввод.
    Возвращает None, если для выбора не удалось собрать варианты (нет correct)."""
    d = data if isinstance(data, dict) else parse_forms(data)
    pos = normalize_pos(d.get("part_of_speech")) or "noun"
    no = row["norwegian"]
    field, label = _FORM_FIELD.get(cell, (cell, cell))
    value = cell_value(pos, forms, cell)
    base = {
        "pool_id": row["pool_id"], "no": no, "translate": d.get("translate", {}),
        # grammar:True — общий флаг тира ★ (фронт рендерит FormPrompt, _attach_choice_options не
        # трогает варианты); form_track:True — ответ маршрутизируется в form_srs, не в overlay/base.
        # repeat:False — слово-то выучено, но это НЕ повтор слова, а изучение формы (без бейджа).
        "part_of_speech": pos, "forms": forms, "form_track": True, "grammar": True,
        "step": cell, "stage": stage, "repeat": False,
        "prompt": {"kind": "lemma+formLabel", "formLabel": label, "lemma": no},
    }
    if stage == "card":                       # показать форму (пассив, как карточка перевода)
        # род: «ei bok»; женское слово валидно и как общего рода (реформа 2005) → учим «ei/en»
        reveal = value
        if cell == "gender":
            reveal = f"ei/en {no}" if value == "ei" else f"{value} {no}"
        return {**base, "mode": "study", "direction": cell,
                "target": {"field": field, "value": value}, "reveal": reveal}
    # gender: и choose, и produce — ВЫБОР артикля (артикль не «печатают»; produce-ступень
    # остаётся в SRS-рампе, но UI тот же выбор — различение en/ei/et и есть продукция рода).
    if stage == "choose" or cell == "gender":
        correct, dis = form_options(pos, no, forms, cell)
        if not correct:
            return None
        if dis:                               # нормальный выбор: верный + правдоподобные подмены
            target = {"field": field, "value": correct}
            if cell == "gender" and correct == "ei":
                target["accept"] = ["en"]     # ei-слово: выбор «en» тоже верен (не наказываем)
            return {**base, "mode": "choice", "direction": cell,
                    "target": target,
                    "options": [{"w": w, "alt": None} for w in [correct] + dis],
                    "distractors": dis}
        # дистракторов не собрать (все правдоподобные варианты — валидные формы слова):
        # однокнопочный «выбор» бессмыслен — отдаём ВВОД, ступень choose всё равно продвинется
    # produce — набрать форму самому
    return {**base, "mode": "input", "direction": cell,
            "target": {"field": field, "value": value}, "scoring": {"typoForgive": False}}


def schedule_form(stage, ease, interval_days, correct, elapsed=None):
    """Чистый шаг планировщика клетки формы (SM-2, как база). Возвращает
    (next_stage, ease, interval_days, due_days). due_days=0 → повтор в ЭТОЙ же сессии.

    Рампа: card (пассив) → choose → produce. Верно двигает ступень; на produce верно → планируем
    повтор (interval*ease). Ошибка → ease вниз, шаг назад, скорый повтор."""
    ease = ease or _EASE_START
    interval_days = interval_days or 0
    idx = FORM_STAGES.index(stage) if stage in FORM_STAGES else 0

    if stage == "card":                        # пассивный показ — сразу к выбору, расписание не трогаем
        return ("choose", ease, interval_days, 0)

    if correct:
        fast = elapsed is not None and elapsed <= _FAST_SEC
        ease = min(_EASE_MAX, ease + (0.08 if fast else 0.04))
        if idx < len(FORM_STAGES) - 1:         # choose→produce: ещё в этой сессии
            return (FORM_STAGES[idx + 1], ease, interval_days, 0)
        interval = 1 if interval_days < 1 else min(_INTERVAL_CAP, round(interval_days * ease))
        if fast and interval < 1:
            interval = 2
        return ("produce", ease, interval, interval)   # клетка отработана → повтор через interval дней

    # ошибка: ease вниз, шаг назад, повтор в этой сессии. interval ОБНУЛЯЕМ (не 1!):
    # «сдана» = interval ≥ 1 — ошибившаяся клетка иначе невидимо считалась сданной
    # (выпадала из счётчика «осталось форм» и из выдачи фазы форм).
    ease = max(_EASE_MIN, ease - 0.2)
    return (FORM_STAGES[max(0, idx - 1)], ease, 0, 0)


# ── DB-слой: form_srs (SRS-состояние клеток форм) ────────────────────────────

# Размер партии цикла «слова↔формы»: столько выученных слов копим, прежде чем
# переключиться на фазу форм. Env-кнопка — для тюнинга без релиза (и тестов).
FORM_CYCLE_BATCH = int(os.getenv("FORM_CYCLE_BATCH", "10"))
FORMS_SESSION_SHARE = 0.7   # доля сессии под формы в фазе forms (остаток — due/прогресс слов)


def _due_in(days):
    return (datetime.utcnow() + timedelta(days=days)).isoformat()


# ── Цикл «слова ↔ формы» (form_cycle) ────────────────────────────────────────

async def _cycle_row(db, user_id):
    async with db.execute("SELECT phase, batch FROM form_cycle WHERE user_id = ?", (user_id,)) as cur:
        r = await cur.fetchone()
    if not r:
        return None
    try:
        batch = [int(x) for x in json.loads(r["batch"] or "[]")]
    except Exception:
        batch = []
    return {"phase": r["phase"] or "words", "batch": batch}


async def _save_cycle(db, user_id, phase, batch):
    await db.execute(
        "INSERT INTO form_cycle (user_id, phase, batch, updated_at) VALUES (?,?,?,?) "
        "ON CONFLICT(user_id) DO UPDATE SET phase=excluded.phase, batch=excluded.batch, "
        "updated_at=excluded.updated_at",
        (user_id, phase, json.dumps(batch), _now()))


async def get_form_cycle(user_id):
    """Состояние цикла юзера или None (ни разу не создавалось — build_session решит сид ветерана)."""
    db = await _conn()
    try:
        return await _cycle_row(db, user_id)
    finally:
        await _release(db)


async def save_form_cycle(user_id, phase, batch):
    db = await _conn()
    try:
        await _save_cycle(db, user_id, phase, batch)
        await db.commit()
    finally:
        await _release(db)


async def note_cycle_mastered(db, user_id, pool_id):
    """Формо-способное слово выучено → в копилку партии (фаза words); FORM_CYCLE_BATCH → фаза forms.
    Вызывает apply_result на переходе в mastered (тем же соединением, коммит — его).
    Выученное во время фазы forms в текущую партию не лезет («пока не выучишь» — про неё) —
    его формы подберёт бэклог-филлер следующих форм-фаз."""
    row = await _cycle_row(db, user_id) or {"phase": "words", "batch": []}
    if row["phase"] != "words":
        return
    batch = row["batch"]
    if pool_id not in batch:
        batch.append(pool_id)
    phase = "forms" if len(batch) >= FORM_CYCLE_BATCH else "words"
    await _save_cycle(db, user_id, phase, batch)


async def load_form_states(user_id, pool_ids=None):
    """Состояния клеток форм юзера: {(pool_id, cell): row}. pool_ids — опц. фильтр (для сессии)."""
    db = await _conn()
    try:
        if pool_ids:
            marks = ",".join("?" * len(pool_ids))
            q = f"SELECT * FROM form_srs WHERE user_id = ? AND pool_id IN ({marks})"
            args = (user_id, *pool_ids)
        else:
            q = "SELECT * FROM form_srs WHERE user_id = ?"
            args = (user_id,)
        async with db.execute(q, args) as cur:
            rows = await cur.fetchall()
        return {(r["pool_id"], r["cell"]): dict(r) for r in rows}
    finally:
        await _release(db)


async def apply_form_result(user_id, pool_id, cell, correct, elapsed=None, stage=None):
    """Ответ по клетке формы → сдвиг рампы/расписания (schedule_form) + дневная активность.

    Как у грамм-overlay: base-SRS слова (user_words) НЕ трогаем — трек форм отдельный слой.
    stage клиента игнорируем в пользу хранимого (анти-рассинхрон двух вкладок); карточка (stage
    'card') активность не пишет (пассивный показ, не ответ)."""
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM form_srs WHERE user_id = ? AND pool_id = ? AND cell = ?",
                              (user_id, pool_id, cell)) as cur:
            r = await cur.fetchone()
        st = dict(r) if r else {"stage": "card", "ease": _EASE_START, "interval_days": 0,
                                "reps": 0, "lapses": 0}
        cur_stage = st.get("stage") or "card"
        nxt, ease, interval, due_days = schedule_form(cur_stage, st.get("ease"), st.get("interval_days"),
                                                      correct, elapsed)
        is_card = cur_stage == "card"
        reps = st["reps"] + (0 if is_card else 1)
        lapses = st["lapses"] + (0 if (is_card or correct) else 1)
        due = _due_in(due_days)
        await db.execute("""
            INSERT INTO form_srs (user_id, pool_id, cell, stage, ease, interval_days, due_at,
                                  reps, lapses, last_seen, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id, pool_id, cell) DO UPDATE SET
                stage=excluded.stage, ease=excluded.ease, interval_days=excluded.interval_days,
                due_at=excluded.due_at, reps=excluded.reps, lapses=excluded.lapses,
                last_seen=excluded.last_seen
        """, (user_id, pool_id, cell, nxt, ease, interval, due, reps, lapses, _now(), _now()))
        if not is_card:   # реальный ответ → стрик/цель/точность (карточка — пассив)
            day = _now()[:10]
            await db.execute("""
                INSERT INTO user_activity (user_id, day, answers, correct) VALUES (?,?,1,?)
                ON CONFLICT(user_id, day) DO UPDATE SET answers = answers + 1, correct = correct + ?
            """, (user_id, day, 1 if correct else 0, 1 if correct else 0))
        await db.commit()
        return {"ok": True, "form": True, "cell": cell, "stage": nxt, "due_at": due}
    finally:
        await _release(db)
