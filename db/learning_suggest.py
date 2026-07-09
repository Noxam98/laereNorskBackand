"""Подсказки/статистика «Учёбы»: сводка прогресса, рабочий уровень, докидывание новых слов/фраз,
добор карточек-знакомств. Вынесено из learning.py.

Публичное: learning_stats (роутер/__init__), suggest_words/next_new_cards (роутер), suggest_phrases/
_phrase_supply/estimate_level (зовёт build_session) — реэкспортируются обратно в learning ПОСЛЕДНИМИ
(после exams/placement, т.к. зависят от них). Сами функции — тонкая надстройка над ядром learning."""
import json

from session.shape import make_element

from .core import _conn, _release, _now
from .learning import (
    LEVELS, LEVEL_TARGETS, _LEVEL_ORDER, PACK, PACK_FIRST, PHRASE_BUFFER, AUDIT_CAP,
    _fetch_user_words, _activity_metrics, _shape, is_function_word,
)
from .exams import _audit_throttled, new_words_blocked
from .placement import get_start_level


def _level_from_total(learned_total, start=None):
    """Достигнутый CEFR-уровень по КУМУЛЯТИВНОМУ числу выученных слов: LEVEL_TARGETS[lv] — сколько
    слов нужно, чтобы БЫТЬ на lv. Берём самый высокий взятый порог, затем флор уровня из плейсмента
    (start). Единый источник для learning_stats и estimate_level — иначе они расходятся."""
    lvl = LEVELS[0]
    for lv in LEVELS:
        if learned_total >= LEVEL_TARGETS[lv]:
            lvl = lv
    if start and _LEVEL_ORDER.get(start, 0) > _LEVEL_ORDER.get(lvl, 0):
        lvl = start
    return lvl


async def learning_stats(user_id):
    db = await _conn()
    try:
        rows = await _fetch_user_words(db, user_id)
        activity = await _activity_metrics(db, user_id)
        throttled = await _audit_throttled(db, user_id)
        # трек ФОРМ: клеток в работе / отработано (produce сдан → interval≥1) / к повторению сейчас
        async with db.execute(
                "SELECT COUNT(*) AS n, "
                "SUM(CASE WHEN interval_days >= 1 THEN 1 ELSE 0 END) AS done, "
                "SUM(CASE WHEN due_at <= ? THEN 1 ELSE 0 END) AS due "
                "FROM form_srs WHERE user_id = ?", (_now(), user_id)) as cur:
            fr = await cur.fetchone()
        forms = {"cells": fr["n"] or 0, "done": fr["done"] or 0, "due": fr["due"] or 0}
    finally:
        await _release(db)
    items = [_shape(r) for r in rows]
    by_status = {}
    by_level = {lv: {"total": 0, "mastered": 0, "target": LEVEL_TARGETS[lv], "done": False} for lv in LEVELS}
    due_count = 0
    for w in items:
        by_status[w["dstatus"]] = by_status.get(w["dstatus"], 0) + 1   # счётчики по ОТОБРАЖАЕМОМУ статусу
        # «к повторению» = только то, что РЕАЛЬНО попадёт в задание: сертифицированные слова
        # повторяются отдельным аудитом (по audit_due), их due_at — вестигиальный (остаётся старым
        # после сертификации) и не должен надувать счётчик «повторить» на карточке (фантомные due).
        if w["due"] and not w["certified"]:
            # выученное СЛУЖЕБНОЕ слово build_session текстовым повтором не обслуживает (у служебных
            # cloze-рампа, due-повтора нет) → его вестигиальный due не идёт в «к повторению» (зеркалит
            # due_mastered из session/pools.py: mastered + due + не серт. + НЕ служебное).
            is_func_mastered = (w["status"] == "mastered"
                                and is_function_word(w["no"], {"part_of_speech": w["part_of_speech"]}))
            if not is_func_mastered:
                due_count += 1
        lv = w["level"] if w["level"] in by_level else None
        if lv:
            by_level[lv]["total"] += 1
            if w["status"] in ("mastered", "archived"):
                by_level[lv]["mastered"] += 1
    # УРОВЕНЬ — КУМУЛЯТИВНЫЙ: LEVEL_TARGETS[lv] = общее число выученных слов, чтобы БЫТЬ на lv.
    # by_level[*].mastered (по CEFR-тегу) остаётся ТОЛЬКО для гистограммы отображения, НЕ для расчёта
    # уровня — прежняя формула (per-tag mastered против кумулятивного target) замораживала currentLevel
    # на плейсмент-тире и запирала юзера в его словаре (сессии сохли). learned_total = всё выученное.
    learned_total = by_status.get("mastered", 0) + by_status.get("repeat", 0) + by_status.get("archived", 0)
    start = await get_start_level(user_id)
    current = _level_from_total(learned_total, start)
    # «done» согласован с currentLevel: уровень закрыт, когда ты его ПРОШЁЛ (текущий выше по порядку),
    # либо это последний тир и добит его порог. Прежняя формула (learned_total >= target) трактовала
    # target как «пройти lv», а _level_from_total — как «достичь lv», поэтому UI на всей полосе показывал
    # «A1 done» и «на A1 → A2» разом. Считаем ПОСЛЕ current (нужен _LEVEL_ORDER[current]).
    for lv in LEVELS:
        by_level[lv]["done"] = (_LEVEL_ORDER[current] > _LEVEL_ORDER[lv]
                                or (lv == LEVELS[-1] and learned_total >= by_level[lv]["target"]))
    # к следующему уровню — первый тир ВЫШЕ текущего, чей кумулятивный порог ещё не взят
    next_goal = next((lv for lv in LEVELS
                      if _LEVEL_ORDER[lv] > _LEVEL_ORDER[current] and learned_total < LEVEL_TARGETS[lv]), None)
    to_next = max(0, LEVEL_TARGETS[next_goal] - learned_total) if next_goal else 0
    # ретеншн = доля выученных среди (выучено+слабых); «выучено за неделю» — по last_seen
    from datetime import datetime, timedelta
    # «выучено» для ретеншна = mastered + repeat (повтор — это тоже выученное, просто подошёл срок) + архив
    mastered_n = learned_total   # то же (mastered+repeat+archived) — считаем один раз для уровня и ретеншна
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
            "retention": retention, "masteredWeek": mastered_week, "gate": gate, "audit": audit,
            "forms": forms}


async def estimate_level(user_id):
    """Рабочий уровень для подсказок — КУМУЛЯТИВНЫЙ (по числу выученных), с флором плейсмента.
    Лёгкий: дешёвый COUNT выученных вместо полного learning_stats (зовётся несколько раз за сборку
    сессии — тяжёлый _fetch_user_words/activity/forms тут не нужен)."""
    db = await _conn()
    try:
        # «Выучено» = mastered (набор клеток рампы пройден ИЛИ «я знаю» через set_status, тоже mastered=1).
        # Голый archived НЕ засчитываем: learning_remove ставит archived=1, mastered=0 (мягкое удаление),
        # иначе после удалений estimate_level расходится с learning_stats.currentLevel (тот считает
        # mastered) и suggest_words подаёт слова труднее, чем показывает карточка уровня.
        async with db.execute(
            "SELECT COUNT(*) AS n FROM user_words "
            "WHERE user_id = ? AND COALESCE(mastered,0) = 1",
            (user_id,)) as cur:
            learned = (await cur.fetchone())["n"]
    finally:
        await _release(db)
    return _level_from_total(learned, await get_start_level(user_id))


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
        # персональная свалка «не учить» — НИКОГДА не предлагать этому юзеру (даже если модератор оставил)
        async with db.execute("SELECT pool_id FROM user_word_skips WHERE user_id = ?", (user_id,)) as cur:
            have |= {r["pool_id"] for r in await cur.fetchall()}
        # «уже знаю» (known) и выученное (mastered) — ЖЁСТКИЙ фильтр: без него suggest заново подсыпал бы
        # слово в скрытый словарь, хотя оно уже known/mastered (расчищенное возвращалось; см. plaсement-baseline).
        async with db.execute(
                "SELECT pool_id FROM user_words WHERE user_id = ? AND (COALESCE(known,0)=1 OR COALESCE(mastered,0)=1)",
                (user_id,)) as cur:
            have |= {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    # кандидаты по уровню, по ЧАСТОТНОСТИ. ВАЖНО: окно расширяем за все уже имеющиеся слова —
    # иначе у юзера с большим словарём топ-N по частоте уже целиком его, новых не находится и пул
    # кажется «исчерпанным» (сессии тают до 1–3 слов, перестают смешиваться).
    window = len(have) + max(count * 6, 60)
    # КУМУЛЯТИВНО (up_to): уровни ≤ lvl по частоте, а не точный тир — иначе placed-юзер заперт в
    # словаре своего тира и после его исчерпания сессии сохнут при нетронутом остальном пуле.
    cand = await pool_by_freq(window, lvl, up_to=True)
    # Фокус на темах: ~1 из 3 кандидатов — из выбранных тем (по частоте), пока тема-слова не кончатся.
    # Пусто → cand без изменений (поведение ровно как раньше). Дедуп — общим циклом ниже (have).
    focus = await get_user_focus_topics(user_id)
    if focus:
        topic_cand = await pool_by_freq_topics(window, lvl, focus, up_to=True)
        if topic_cand:
            merged, ti, ni = [], 0, 0
            while ti < len(topic_cand) or ni < len(cand):
                if ti < len(topic_cand):
                    merged.append(topic_cand[ti]); ti += 1
                for _ in range(2):
                    if ni < len(cand):
                        merged.append(cand[ni]); ni += 1
            cand = merged
    # Компаунд-приоритет (session/compounds): рычажные корни раньше собранных из них слов,
    # композиты мягко придержаны, комплементы известных основ — вперёд. Только ПЕРЕупорядочивает
    # (ничего не выбрасывает); индекс пуст / сбой → порядок как был (частотный).
    try:
        from session import compounds as _comp
        from .compound_index import load_index, mastered_set as _mastered_set
        index, feat, learnable = await load_index()
        if index and cand:
            dbm = await _conn()
            try:
                mastered = await _mastered_set(dbm, user_id)
            finally:
                await _release(dbm)
            order = _comp.rank_new([(w["norwegian"], w.get("freq") or 0) for w in cand],
                                   feat=feat, mastered=mastered, learnable=learnable)
            rank = {w: i for i, w in enumerate(order)}
            cand.sort(key=lambda w: rank.get(w["norwegian"], 1 << 30))
    except Exception:
        from config import logger as _lg
        _lg.warning("suggest_words: компаунд-ранжирование пропущено", exc_info=True)
    added = []
    # Целевая ПОЛОСА вокруг уровня (i+1), а не потолок: большинство — НА уровне; ~20% — СТРЕТЧ L+1
    # (растущий край; раньше недостижим — up_to смотрел только ≤L, и placed-B1 никогда не видел B2);
    # ниже уровня («добор пробелов») — ~20% (known/mastered уже отфильтрованы в have). Частота сортирует
    # ВНУТРИ полосы. A1-новичка не режем (li==0, ниже пусто). Нехватка → снимаем капы, потом fallthrough.
    li = LEVELS.index(lvl) if lvl in LEVELS else 0
    below_cap = count if li == 0 else max(1, round(count * 0.2))
    has_stretch = li + 1 < len(LEVELS)
    stretch_cap = round(count * 0.2) if has_stretch else 0
    stretch = await pool_by_freq(window, LEVELS[li + 1], up_to=False) if has_stretch else []
    below_added = 0

    def _below_level(w):
        wl = w.get("level")
        return bool(wl) and wl in LEVELS and LEVELS.index(wl) < li

    async def _take(cands, cap=None, respect_band=True):
        nonlocal below_added
        stop = cap if cap is not None else count
        for w in cands:
            if len(added) >= stop:
                return
            pid = w["pool_id"]
            if not pid or pid in have:
                continue
            # гейт A1: пока нет базы контентных — служебные не досыпаем (иначе «новый пул» забьётся ими)
            if not allow_func and is_function_word(w["norwegian"], {"part_of_speech": w.get("part_of_speech")}):
                continue
            below = _below_level(w)
            if respect_band and below and below_added >= below_cap:
                continue   # квота слов ниже уровня исчерпана — ждём слова своего уровня
            res = await add_word_to_dict(user_id, target_id, pid)
            if res.get("id") and not res.get("duplicate"):
                added.append({"pool_id": pid, "no": w["norwegian"], "translate": w.get("translate", {})})
                have.add(pid)
                if below:
                    below_added += 1

    # ≤L (в основном НА уровне + ≤below_cap ниже), оставляя место под стретч
    await _take(cand, cap=count - stretch_cap)
    if stretch:                              # +СТРЕТЧ L+1 (растущий край) — до полной порции
        await _take(stretch, respect_band=False)
    if len(added) < count:                   # стретча/уровня не хватило → добираем ≤L без капа
        await _take(cand, respect_band=False)
    # Fallthrough: всё ≤L+1 уже в словаре юзера — добираем ЛЮБЫМ уровнем по частоте (сессия не сохнет).
    if len(added) < count:
        await _take(await pool_by_freq(len(have) + max(count * 6, 60), None), respect_band=False)
    return {"added": len(added), "words": added, "level": lvl, "dict": "__auto__"}


COMPOUND_BUFFER = 6   # столько ОТКРЫТЫХ (из выученных основ) композитов держим доступными за раз


async def suggest_compounds(user_id, count=COMPOUND_BUFFER):
    """Подмешать составные слова, ОТКРЫТЫЕ выученными основами (обе части mastered, слова у юзера
    ещё нет), в скрытый авто-словарь — зеркало suggest_phrases. Это ДОБАВКА к частотному потоку
    (не гейт): унлок как награда за основы. Порядок — по частоте (употребимые раньше)."""
    from .dictionaries import add_word_to_dict, get_or_create_hidden_dict
    from session import compounds as _comp
    from .compound_index import load_index
    if await new_words_blocked(user_id):
        return {"added": 0}
    index, _feat, _learnable = await load_index()
    if not index:
        return {"added": 0}
    db = await _conn()
    try:
        async with db.execute(
                "SELECT w.norwegian FROM user_words uw JOIN word_pool w ON w.id = uw.pool_id "
                "WHERE uw.user_id = ? AND uw.mastered = 1", (user_id,)) as cur:
            mastered = {r["norwegian"] for r in await cur.fetchall()}
        # have — pool_id (а не написание): у омонима одна запись не должна отсекать вторую
        async with db.execute(
                "SELECT dw.pool_id FROM dict_words dw JOIN dictionaries d ON d.id = dw.dict_id "
                "WHERE d.user_id = ?", (user_id,)) as cur:
            have = {r["pool_id"] for r in await cur.fetchall()}
        async with db.execute(   # known/mastered тоже исключаем (parity с suggest_words/phrases)
                "SELECT pool_id FROM user_words WHERE user_id = ? AND (COALESCE(known,0)=1 OR COALESCE(mastered,0)=1)",
                (user_id,)) as cur:
            have |= {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    unlocked = _comp.eligible_unlocks(index, mastered, have)
    if not unlocked:
        return {"added": 0}
    unlocked.sort(key=lambda c: -(c.get("freq") or 0))
    hid = await get_or_create_hidden_dict(user_id)
    added = 0
    for c in unlocked[:count]:
        res = await add_word_to_dict(user_id, hid, c["pool_id"])
        if res.get("id") and not res.get("duplicate"):
            added += 1
    return {"added": added}


async def unlocked_compounds_count(user_id):
    """Сколько составных слов открыто выученными основами, но ещё не в словаре юзера (для «Сегодня»)."""
    from session import compounds as _comp
    from .compound_index import load_index
    index, _feat, _learnable = await load_index()
    if not index:
        return 0
    db = await _conn()
    try:
        async with db.execute(
                "SELECT w.norwegian FROM user_words uw JOIN word_pool w ON w.id = uw.pool_id "
                "WHERE uw.user_id = ? AND uw.mastered = 1", (user_id,)) as cur:
            mastered = {r["norwegian"] for r in await cur.fetchall()}
        # have — pool_id (а не написание): у омонима одна запись не должна отсекать вторую
        async with db.execute(
                "SELECT dw.pool_id FROM dict_words dw JOIN dictionaries d ON d.id = dw.dict_id "
                "WHERE d.user_id = ?", (user_id,)) as cur:
            have = {r["pool_id"] for r in await cur.fetchall()}
    finally:
        await _release(db)
    return len(_comp.eligible_unlocks(index, mastered, have))


async def _phrase_supply(db, user_id):
    """Сколько у юзера ещё НЕ начатых устойчивых выражений в Учёбе (studying-словари, 0 попыток,
    не архив/не «знакомые») — для троттлинга подмешивания фраз."""
    async with db.execute(
        """SELECT COUNT(*) c FROM dict_words dw
           JOIN dictionaries d ON d.id = dw.dict_id AND d.user_id = ? AND COALESCE(d.studying,1) = 1
           JOIN word_pool wp ON wp.id = dw.pool_id AND wp.pos = 'phrase'
           LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
           WHERE COALESCE(uw.correct,0) + COALESCE(uw.incorrect,0) = 0
             AND COALESCE(uw.archived,0) = 0 AND COALESCE(uw.known,0) = 0""",
        (user_id, user_id)) as cur:
        return (await cur.fetchone())["c"]


async def suggest_phrases(user_id, count=PHRASE_BUFFER, level=None):
    """Подмешать НОВЫЕ устойчивые выражения (pos='phrase') в Учёбу — в СКРЫТЫЙ авто-словарь
    (studying=1), как suggest_words. КУМУЛЯТИВНО по уровню: фразы ВСЕХ уровней ≤ уровня юзера
    (это словосочетания, а не элементарные слова — младшие уместны и на старших уровнях). Только
    game-ready (есть data.game.distractors ≥2 — иначе order-игра пустая), которых у юзера ещё нет.
    Возвращает {added}. Приток закрыт воротами экзамена → ничего не добавляем."""
    from .dictionaries import add_word_to_dict, get_or_create_hidden_dict
    if await new_words_blocked(user_id):
        return {"added": 0, "blocked": True}
    lvl = level if level in LEVELS else await estimate_level(user_id)
    allowed = LEVELS[:LEVELS.index(lvl) + 1] if lvl in LEVELS else LEVELS[:1]   # A1..lvl
    target_id = await get_or_create_hidden_dict(user_id)
    db = await _conn()
    try:
        async with db.execute("SELECT DISTINCT pool_id FROM dict_words WHERE dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)", (user_id,)) as cur:
            have = {r["pool_id"] for r in await cur.fetchall()}
        async with db.execute("SELECT pool_id FROM user_word_skips WHERE user_id = ?", (user_id,)) as cur:
            have |= {r["pool_id"] for r in await cur.fetchall()}
        async with db.execute(   # known/mastered — жёсткий фильтр (иначе известные фразы «se på tv» подсыпались заново)
                "SELECT pool_id FROM user_words WHERE user_id = ? AND (COALESCE(known,0)=1 OR COALESCE(mastered,0)=1)",
                (user_id,)) as cur:
            have |= {r["pool_id"] for r in await cur.fetchall()}
        marks = ",".join("?" for _ in allowed)
        # 'A1'<'A2'<'B1'… лексикографически; DESC = БЛИЖЕ К УРОВНЮ первыми (B1-юзеру B1-фразы, а не A1-завал).
        # Младшие фразы допустимы (словосочетания), но идут В ХВОСТЕ, а не заваливают продвинутого.
        async with db.execute(
            f"SELECT id, norwegian, data FROM word_pool "
            f"WHERE pos='phrase' AND COALESCE(learn_excluded,0)=0 AND level IN ({marks}) "
            f"ORDER BY level DESC, id", allowed) as cur:
            cands = [dict(r) for r in await cur.fetchall()]
    finally:
        await _release(db)
    added = 0
    for c in cands:
        if added >= count:
            break
        pid = c["id"]
        if pid in have:
            continue
        try:
            g = ((json.loads(c["data"]) if c["data"] else {}).get("game") or {}).get("distractors") or []
        except Exception:
            g = []
        if len(g) < 2:                    # без дистракторов order-игра пустая — пропускаем
            continue
        res = await add_word_to_dict(user_id, target_id, pid)
        if res.get("id") and not res.get("duplicate"):
            added += 1
            have.add(pid)
    return {"added": added}


async def _own_new_card_rows(user_id, exclude):
    """СОБСТВЕННЫЕ ещё не начатые новые слова юзера (как их берёт build_session): из словарей «в
    обучении», 0 попыток, не архив/не «знакомые». По частоте. exclude — pool_id уже в очереди."""
    db = await _conn()
    try:
        async with db.execute(
            """SELECT wp.id AS pool_id, wp.norwegian, wp.data, wp.forms
               FROM dict_words dw
               JOIN dictionaries d ON d.id = dw.dict_id AND d.user_id = ? AND COALESCE(d.studying, 1) = 1
               JOIN word_pool wp ON wp.id = dw.pool_id
               LEFT JOIN user_words uw ON uw.user_id = ? AND uw.pool_id = wp.id
               WHERE COALESCE(uw.correct,0) + COALESCE(uw.incorrect,0) = 0
                 AND COALESCE(uw.archived,0) = 0 AND COALESCE(uw.known,0) = 0
                 AND COALESCE(wp.learn_excluded,0) = 0
               GROUP BY wp.id
               ORDER BY wp.freq IS NULL, wp.freq DESC""", (user_id, user_id)) as cur:
            return [r for r in await cur.fetchall() if r["pool_id"] not in exclude]
    finally:
        await _release(db)


async def next_new_cards(user_id, n=5, exclude=None):
    """Живая сессия: добор НОВЫХ карточек-знакомств по требованию (когда юзер убрал карточку кнопкой,
    а не «принял» тыком). СНАЧАЛА отдаёт СОБСТВЕННЫЕ ещё не начатые новые слова юзера (их может быть
    много — он ими владеет, просто не дошёл), и лишь если своих не хватило — досыпает из общего пула
    через suggest_words. exclude — pool_id, уже стоящие в очереди. Карточки той же формы, что
    build_session. {"cards": [...]} либо {"cards": [], "blocked": True}, если приток новых закрыт воротами."""
    exclude = set(exclude or [])
    n = max(1, min(int(n or 1), 20))
    if await new_words_blocked(user_id):          # ворота экзамена / тормоз аудита — новые не вводим
        return {"cards": [], "blocked": True}
    rows = await _own_new_card_rows(user_id, exclude)
    if len(rows) < n:                             # своих не хватило → досыпаем из пула и перечитываем
        # allow_func=False: добор-карточки середины сессии — только контентные; служебные вводит
        # build_session со своим пословным гейтом (иначе тут они просачивались бы мимо A1-гейта).
        await suggest_words(user_id, count=(n - len(rows)) + len(exclude) + 4, allow_func=False)
        rows = await _own_new_card_rows(user_id, exclude)
    if not rows:
        return {"cards": []}
    cards = []
    for r in rows[:n]:
        data = json.loads(r["data"]) if r["data"] else {}
        try:
            _forms = json.loads(r["forms"]) if r["forms"] else None   # битый forms не роняет добор
        except Exception:
            _forms = None
        cards.append(make_element(
            pool_id=r["pool_id"], no=r["norwegian"],
            translate=data.get("translate", {}),
            part_of_speech=data.get("part_of_speech", ""),
            gloss=data.get("gloss"), example=data.get("example"),
            forms=_forms,
            mode="study", direction=None, step="card", repeat=False,
        ))
    return {"cards": cards}
