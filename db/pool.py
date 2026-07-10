import json
import os
import time
import asyncio
import fuzzy
import numpy as np
from rapidfuzz.process import cdist as rf_cdist
from rapidfuzz.distance import OSA
from .core import _conn, _release, _now, normalize_word, vec_upsert, vec_delete, vec_nearest_rows
from .pool_queues import has_tts_expr, no_tts_expr   # единый источник правды «есть озвучка»
from langs import LANG_SET
from pos import normalize_pos, POS_KEYS


def _fold_no(s):
    """Свернуть норвежские å/ø/æ → a/o/ae (для поиска без норвежской раскладки: «male»↔«måle»)."""
    return (s or "").replace("å", "a").replace("ø", "o").replace("æ", "ae")


# SQL-выражение, складывающее å/ø/æ в колонке norwegian (для LIKE без учёта норв. букв)
_SQL_FOLD_NO = "replace(replace(replace(norwegian,'å','a'),'ø','o'),'æ','ae')"

_POOL_LANGS = LANG_SET  # языки интерфейса = ключи translate в data (из реестра langs.py)
_SEM_MAX_DIST = float(os.getenv("POOL_SEM_MAX_DIST", "0.52"))   # порог косинус-дистанции для семантики
_SEM_MAX_RESULTS = int(os.getenv("POOL_SEM_MAX_RESULTS", "12"))  # семантика — подсказка, не полный список


def _key_cond_with_forms(key, lang):
    """_key_cond + сведение словоформы к леммам банка (gikk → gå): грид Базы находит
    слово по любой его форме."""
    c, p = _key_cond(key, lang)
    try:
        from db import ordbank
        lemmas = sorted({w for w, _pos in ordbank.exact_form(key)})
    except Exception:
        lemmas = []
    if lemmas:
        marks = ",".join("?" for _ in lemmas)
        c = c[:-1] + f" OR norwegian IN ({marks}))"
        p = p + lemmas
    return c, p


def _key_cond(key, lang):
    """SQL-условие подстрочного поиска: норвежский (+ свёртка å/ø/æ) И перевод на язык
    интерфейса (json_extract по translate.<lang>). Ищем только по двум языкам — норвежскому и
    системному — чтобы не цеплять чужие языки (напр. «male» в польском «maleć»). Если lang не из
    списка — fallback на data LIKE (все языки, прежнее поведение). Возвращает (cond, params)."""
    parts = ["norwegian LIKE ?", f"{_SQL_FOLD_NO} LIKE ?"]
    params = [f"%{key}%", f"%{_fold_no(key)}%"]
    if lang in _POOL_LANGS:
        parts.append(f"json_extract(data, '$.translate.{lang}') LIKE ?")
    else:
        parts.append("data LIKE ?")
    params.append(f"%{key}%")
    return "(" + " OR ".join(parts) + ")", params


def _loads(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def _pool_relevance(norwegian, data, qn, qf, freq, lang):
    """Оценка релевантности слова запросу (выше = вероятнее искомое). Сигналы: точное/префиксное/
    подстрочное совпадение норвежского (со свёрткой å/ø/æ) и перевода на язык интерфейса целым
    словом/префиксом/подстрокой; частота — лёгкий тай-брейкер."""
    nwf = _fold_no((norwegian or "").lower())
    s = 0
    if qf and nwf == qf:
        s = 1000
    elif qf and nwf.startswith(qf):
        s = 850 - min(len(nwf) - len(qf), 50)
    elif qf and qf in nwf:
        s = 600 - min(nwf.index(qf), 50)
    tr = data.get("translate") or {}
    arr = tr.get(lang) if lang in _POOL_LANGS else None
    if not isinstance(arr, list):  # без языка интерфейса — по всем переводам
        arr = [t for v in tr.values() if isinstance(v, list) for t in v]
    for t in (arr or []):
        tl = (t or "").lower()
        if not tl:
            continue
        if tl == qn:
            s = max(s, 800)
        elif qn in tl.split():
            s = max(s, 750)
        elif tl.startswith(qn):
            s = max(s, 500)
        elif qn in tl:
            s = max(s, 200)
    return s + min(float(freq or 0), 8.0) * 0.5


# ---- Нечёткий (fuzzy) поиск по пулу: индекс токенов в памяти + rapidfuzz (C++) ----
# Полный скан с json.loads+normalize по 6к слов дорог (~2-5с на слабом CPU). Строим индекс
# ОДИН раз (плоский список токенов + numpy-массивы pool_id/допуск) и переиспользуем; матчинг
# запроса считает rapidfuzz.process.cdist (весь проход в C++ одним вызовом, ~десятки мс).
# Инвалидация: по числу слов (новое сгенерили → пересоберём в фоне) + TTL (ловит правки).
_FUZZY_IDX = {"count": -1, "built": 0.0, "flat": None}  # flat: (tokens[list], pids[np], tols[np])
_FUZZY_TTL = 600  # сек


def _tok_tol(n):
    """Допуск правок для токена длины n (как fuzzy.tol_for): ≤3 — 0, 4–7 — 1, 8+ — 2."""
    return 0 if n <= 3 else 1 if n <= 7 else 2


def _build_fuzzy_flat(fetched):
    """CPU-часть постройки индекса (json.loads + normalize) — гоняется в потоке.
    Возвращает (tokens[list[str]], pids[np.int64], tols[np.uint8])."""
    tokens, pids, tols = [], [], []
    for r in fetched:
        try:
            d = json.loads(r["data"]) if r["data"] else {}
        except Exception:
            d = {}
        terms = [r["norwegian"]]
        for arr in (d.get("translate") or {}).values():
            if isinstance(arr, list):
                terms.extend(arr)
        seen = set()
        for t in terms:
            for w in fuzzy.normalize(t).split():
                if len(w) >= 2 and w not in seen:
                    seen.add(w)
                    tokens.append(w)
                    pids.append(r["id"])
                    tols.append(_tok_tol(len(w)))
    return tokens, np.asarray(pids, dtype=np.int64), np.asarray(tols, dtype=np.uint8)


_FUZZY_REBUILDING = {"on": False}


async def _rebuild_fuzzy_index(cnt, db=None):
    own = db is None
    if own:
        db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian, data FROM word_pool") as cur:
            fetched = await cur.fetchall()
    finally:
        if own:
            await _release(db)
    flat = await asyncio.to_thread(_build_fuzzy_flat, fetched)
    _FUZZY_IDX.update(count=cnt, built=time.time(), flat=flat)


async def _ensure_fuzzy_index(db):
    async with db.execute("SELECT COUNT(*) c FROM word_pool") as cur:
        cnt = (await cur.fetchone())["c"]
    if _FUZZY_IDX["flat"] is not None and _FUZZY_IDX["count"] == cnt and (time.time() - _FUZZY_IDX["built"]) < _FUZZY_TTL:
        return
    if _FUZZY_IDX["flat"] is not None:
        # есть устаревший индекс — отдаём его сразу, пересборка в фоне (не блокирует этот запрос)
        if not _FUZZY_REBUILDING["on"]:
            _FUZZY_REBUILDING["on"] = True
            async def _bg():
                try:
                    await _rebuild_fuzzy_index(cnt)
                finally:
                    _FUZZY_REBUILDING["on"] = False
            asyncio.create_task(_bg())
        return
    # индекса ещё нет — строим синхронно один раз (на том же соединении)
    await _rebuild_fuzzy_index(cnt, db)


def _fuzzy_scan(qn, flat, limit):
    """Матчинг запроса по кэш-индексу через rapidfuzz (C++ батч): [pool_id] по росту расстояния."""
    tokens, pids, tols = flat
    if not tokens:
        return []
    # OSA-расстояние запрос × все токены одним вызовом C++; >2 отсекаются (score_cutoff)
    row = rf_cdist([qn], tokens, scorer=OSA.distance, score_cutoff=2, dtype=np.uint8)[0]
    idx = np.nonzero(row <= tols)[0]            # принятые: расстояние ≤ допуска токена
    if idx.size == 0:
        return []
    best = {}
    for i in idx.tolist():
        pid = int(pids[i]); dd = int(row[i])
        if best.get(pid, 99) > dd:
            best[pid] = dd
    return [pid for pid, _ in sorted(best.items(), key=lambda kv: kv[1])[:limit]]


async def fuzzy_pool_ids(db, query, limit=10):
    """Ранжированные pool_id по неточному совпадению (норвежский + переводы любых языков),
    через кэш-индекс + rapidfuzz. CPU-проход — в отдельном потоке (не блокирует event loop)."""
    qn = fuzzy.normalize(query or "")
    if len(qn) < 3:
        return []
    await _ensure_fuzzy_index(db)
    return await asyncio.to_thread(_fuzzy_scan, qn, _FUZZY_IDX["flat"], limit)


async def _lemma_redirect(db, key, pos):
    """Если новое слово `key` (нормализованное) — ФОРМА уже существующей леммы (значится в её колонке
    forms), вернуть id этой леммы: привязываем юзера к лемме, а дубль-форму не создаём. Только при
    ОДНОЗНАЧНОМ совпадении (ровно одна подходящая лемма) и совместимом pos — иначе None (создаём как
    обычно). LIKE — грубый префильтр, дальше точная сверка по распарсенным формам (исключаем подстроки)."""
    cands = []
    async with db.execute(
        "SELECT id, norwegian, forms, COALESCE(pos,'') p FROM word_pool WHERE forms LIKE ?",
        (f"%{key}%",)) as cur:
        rows = await cur.fetchall()
    for r in rows:
        try:
            f = json.loads(r["forms"]) if r["forms"] else {}
        except Exception:
            continue
        vals = set()
        for k2, v in f.items():
            if k2 in ("pos", "gender") or not isinstance(v, str):
                continue
            for part in v.replace("har ", "").replace("er ", "").split("/"):
                vals.add(part.strip().lower())
        if key in vals and (r["norwegian"] or "").strip().lower() != key:
            if pos and r["p"] and r["p"] != pos:   # часть речи несовместима — не редиректим
                continue
            cands.append(r["id"])
    return cands[0] if len(cands) == 1 else None


async def get_or_create_pool(norwegian: str, data: dict, created_by: int = None, approved: int = 1):
    """Вернуть id записи пула для (норвежское слово + часть речи), создав её при необходимости.
    Запись определяется парой (norwegian, pos) — омонимы (føde «еда»/«рожать») = разные записи.
    created_by/approved задаются только для НОВОЙ записи (существующую не трогаем — иначе можно
    случайно «разодобрить» общее слово). approved=0 + created_by=user → личное расширение."""
    key = normalize_word(norwegian)
    if not key:
        return None
    pos = normalize_pos((data or {}).get("part_of_speech"))   # норв.↔англ. → канон (substantiv→noun…)
    if data and data.get("part_of_speech") and data["part_of_speech"] != pos:
        data = {**data, "part_of_speech": pos}                # храним каноничный POS и в data
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ?", (key, pos)) as cur:
            row = await cur.fetchone()
            if row:
                return row["id"]
        # новое слово: если это ФОРМА существующей леммы (dager→dag, gir→gi) — привязываем к лемме,
        # дубль-форму не создаём (юзер всё равно получает слово — лемму из базы)
        lemma_id = await _lemma_redirect(db, key, pos)
        if lemma_id:
            return lemma_id
        # частотность проставляем СРАЗУ при создании — из корпус-лексикона (нет в нём → 0.0)
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, created_at, pos, created_by, approved, freq) "
            "VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT zipf FROM nb_lexicon WHERE word = ?), 0.0))",
            (key, json.dumps(data, ensure_ascii=False), _now(), pos, created_by, approved, key),
        )
        await db.commit()
        try:                          # новое пул-слово: сбросить TTL-кеш compound-индекса
            from .compound_index import invalidate   # (learnable/дистракторы зависят от всего пула)
            invalidate()
        except Exception:
            pass
        return cur.lastrowid
    finally:
        await _release(db)


async def get_pool_id(norwegian: str, pos: str = None):
    """id записи пула по норвежскому слову (без создания). Если задан pos — по паре
    (norwegian, pos) точно; иначе любая запись с этим словом (старшая по id)."""
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        if pos is not None:
            sql, args = "SELECT id FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ?", (key, pos or "")
        else:
            sql, args = "SELECT id FROM word_pool WHERE norwegian = ? ORDER BY id LIMIT 1", (key,)
        async with db.execute(sql, args) as cur:
            r = await cur.fetchone()
            return r["id"] if r else None
    finally:
        await _release(db)


async def get_pool_by_id(pool_id: int, user_id: int = None):
    """Запись пула по id. Видимость модерации применяется ТОЛЬКО когда передан user_id (контекст
    запроса пользователя): неодобренное (approved=0) слово читается лишь его автором (created_by ==
    user_id) — иначе перебором id читались бы чужие pending-слова (/pool/{word}/meta|description|
    synonyms|distractors). БЕЗ user_id (внутренние/фоновые вызовы — эмбеддинги, автофилл) фильтра нет."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT * FROM word_pool WHERE id = ? AND (COALESCE(approved,1) = 1 OR created_by = ? OR ? IS NULL)",
            (pool_id, user_id, user_id),
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "data": json.loads(row["data"]),
                "description": json.loads(row["description"]) if row["description"] else None,
                "embedding": row["embedding"],  # сырые байты
                "forms": json.loads(row["forms"]) if row["forms"] else None,
                "created_by": row["created_by"],   # автор (NULL = системное/общая база)
                "approved": row["approved"],        # 1 общая база / 0 pending / 2 rejected
            }
    finally:
        await _release(db)







# Частотность (Zipf) вынесена в db/pool_freq.py (реэкспорт в конце файла).


# Дедуп/слияние вынесены в db/pool_dedup.py (реэкспорт в конце файла).
async def get_pool_meta_all():
    """[[id, norwegian, {lng:tr}, pop]] — лёгкие метаданные пула для дедупа (без векторов)."""
    db = await _conn()
    try:
        pop = {}
        async with db.execute("SELECT pool_id, COUNT(*) c FROM dict_words GROUP BY pool_id") as cur:
            for r in await cur.fetchall():
                pop[r["pool_id"]] = r["c"]
        out = []
        async with db.execute("SELECT id, norwegian, data FROM word_pool WHERE embedding IS NOT NULL") as cur:
            for r in await cur.fetchall():
                try:
                    t = (json.loads(r["data"]) if r["data"] else {}).get("translate", {}) or {}
                except Exception:
                    t = {}
                tr = {lng: t[lng][0] for lng in ("ru", "en", "ukr") if t.get(lng)}
                out.append([r["id"], r["norwegian"], tr, pop.get(r["id"], 0)])
        return out
    finally:
        await _release(db)






async def update_pool_word(old_norwegian: str, translate: dict):
    """Правка слова в ОБЩЕМ пуле (для всех): обновить переводы (любые из ru/ukr/en/pl/lt/no)
    и, если изменилось норвежское слово (translate['no']), — переименовать (norwegian-ключ +
    data.word/translate.no). Сбрасывает emb_sem (пере-эмбеддинг по смыслу) и tts_tr_done.
    Возвращает {ok, norwegian} либо {error: not_found|exists}."""
    key = normalize_word(old_norwegian)
    db = await _conn()
    renamed = False
    try:
        async with db.execute("SELECT id, data FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "not_found"}
        pid = row["id"]
        data = json.loads(row["data"]) if row["data"] else {}
        tr = dict(data.get("translate", {}) or {})
        for k, v in (translate or {}).items():
            if isinstance(v, list) and v:
                tr[k] = v
        data["translate"] = tr
        new_no = (tr.get("no") or [None])[0]
        new_key = normalize_word(new_no) if new_no else key
        if new_key != key:
            async with db.execute("SELECT 1 FROM word_pool WHERE norwegian = ? AND id != ?", (new_key, pid)) as c2:
                if await c2.fetchone():
                    return {"error": "exists"}
            data["word"] = new_no
        await db.execute(
            # embedding=NULL: иначе на рестарте до пере-эмбеддинга load_pool_embeddings (WHERE
            # embedding IS NOT NULL) вернёт stale-вектор в кеш дистракторов (как replace_pool_word)
            "UPDATE word_pool SET data = ?, norwegian = ?, embedding = NULL, emb_sem = 0, "
            "tts_tr_done = 0, translate_done = 1 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), new_key, pid),
        )
        await db.commit()
        renamed = new_key != key
        try:                          # смысл слова изменился (emb_sem=0 → пере-эмбеддинг): убрать
            import embcache           # stale-вектор из резидентного кеша (дистракторы), консистентно
            embcache.remove_vec(pid)  # с delete/merge; reembed пере-добавит свежий через update_vec
        except Exception:
            pass
    finally:
        await _release(db)
    _invalidate_candidates()          # правка не должна светиться в кеше «похожих»/дистракторов
    if renamed:                       # переименование → инвалидируем и compound-индекс (как delete)
        try:
            from .compound_index import invalidate
            invalidate()
        except Exception:
            pass
    return {"ok": True, "norwegian": new_key}


async def replace_pool_word(old_norwegian: str, new_norwegian: str, data: dict,
                            pos: str = None, pool_id: int = None):
    """Заменить слово в пуле стандартизованными данными (исправленное норв. слово + переводы +
    часть речи) и СБРОСИТЬ производное (эмбеддинг/формы/озвучку/флаги) — фон пересоздаст их
    заново для нового написания. pool_id сохраняется (ссылки словарей не рвутся).
    Адресуем КОНКРЕТНУЮ запись омонима: по pool_id (точно), иначе по (norwegian, pos), иначе —
    старшая по norwegian (обратная совместимость). Колонку pos держим в синхроне с
    data.part_of_speech (иначе рассинхрон ключа (norwegian,pos) → дубли в get_or_create_pool).
    Возвращает {ok, norwegian} либо {error: not_found|exists}."""
    old_key = normalize_word(old_norwegian)
    new_key = normalize_word(new_norwegian) or old_key
    new_pos = normalize_pos((data or {}).get("part_of_speech"))
    db = await _conn()
    renamed = False
    try:
        if pool_id is not None:
            sel_sql = "SELECT id, norwegian, COALESCE(pos,'') p FROM word_pool WHERE id = ?"
            sel_args = (pool_id,)
        elif pos is not None:
            sel_sql = "SELECT id, norwegian, COALESCE(pos,'') p FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ?"
            sel_args = (old_key, pos or "")
        else:
            sel_sql = "SELECT id, norwegian, COALESCE(pos,'') p FROM word_pool WHERE norwegian = ? ORDER BY id LIMIT 1"
            sel_args = (old_key,)
        async with db.execute(sel_sql, sel_args) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "not_found"}
        pid = row["id"]
        old_row_key, old_row_pos = row["norwegian"], row["p"]
        # коллизия — по ПАРЕ (новое слово, новая часть речи): омонимы (одно слово, разные pos) легальны
        if new_key != old_row_key or new_pos != old_row_pos:
            async with db.execute(
                "SELECT 1 FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ? AND id != ?",
                (new_key, new_pos, pid)) as c2:
                if await c2.fetchone():
                    return {"error": "exists"}
        await db.execute(
            "UPDATE word_pool SET data = ?, norwegian = ?, pos = ?, embedding = NULL, emb_sem = 0, "
            "forms = NULL, tts_tr_done = 0, translate_done = 0 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), new_key, new_pos, pid),
        )
        await db.commit()
        renamed = new_key != old_row_key
        try:                          # embedding занулён → убрать stale-вектор из резидентного кеша
            import embcache           # (дистракторы), консистентно с delete/merge; reembed пере-
            embcache.remove_vec(pid)  # добавит свежий через set_pool_embedding→update_vec
        except Exception:
            pass
    finally:
        await _release(db)
    _invalidate_candidates()          # правка не должна светиться в кеше «похожих»/дистракторов
    if renamed:                       # переименование → инвалидируем и compound-индекс (как delete)
        try:
            from .compound_index import invalidate
            invalidate()
        except Exception:
            pass
    return {"ok": True, "norwegian": new_key}


async def update_pool_translate(pool_id: int, translate: dict):
    """Записать обновлённый словарь translate в data слова; сбросить tts_tr_done,
    чтобы фон озвучил новые языки."""
    db = await _conn()
    try:
        async with db.execute("SELECT data FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return
        data = json.loads(row["data"])
        data["translate"] = translate
        await db.execute(
            "UPDATE word_pool SET data = ?, tts_tr_done = 0 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), pool_id),
        )
        await db.commit()
    finally:
        await _release(db)








async def get_pool_sample(limit: int = 120):
    """Случайная выборка норвежских слов из пула (исключения для генерации по теме)."""
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool ORDER BY RANDOM() LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_duel_words(limit: int = 80, level: str = None, topic: str = None):
    """Случайные слова пула с переводами (для онлайн-игр). Фильтры: уровень CEFR, тема.
    [{norwegian, translate{lang:[...]}, part_of_speech}]."""
    conds, params = ["data IS NOT NULL"], []
    if level:
        conds.append("level = ?"); params.append(level)
    if topic:
        conds.append("id IN (SELECT pool_id FROM word_topics WHERE topic = ?)"); params.append(topic)
    sql = f"SELECT norwegian, data, embedding FROM word_pool WHERE {' AND '.join(conds)} ORDER BY RANDOM() LIMIT ?"
    params.append(limit)
    db = await _conn()
    try:
        async with db.execute(sql, params) as cur:
            out = []
            for r in await cur.fetchall():
                try:
                    d = json.loads(r["data"]) or {}
                except Exception:
                    d = {}
                tr = d.get("translate", {}) or {}
                if tr:
                    out.append({"norwegian": r["norwegian"], "translate": tr,
                                "part_of_speech": d.get("part_of_speech", ""),
                                "embedding": r["embedding"]})
            return out
    finally:
        await _release(db)


async def get_pool_words_by_names(names):
    """Слова пула по списку норвежских имён: [{norwegian, translate, embedding}] (для AI-набора)."""
    if not names:
        return []
    marks = ",".join("?" for _ in names)
    db = await _conn()
    try:
        async with db.execute(
            f"SELECT norwegian, data, embedding FROM word_pool WHERE norwegian IN ({marks})", list(names)
        ) as cur:
            out = []
            for r in await cur.fetchall():
                try:
                    tr = (json.loads(r["data"]) or {}).get("translate", {}) or {}
                except Exception:
                    tr = {}
                if tr:
                    out.append({"norwegian": r["norwegian"], "translate": tr, "embedding": r["embedding"]})
            return out
    finally:
        await _release(db)


async def get_pool_letter(letter: str, limit: int = 120):
    """Норвежские слова пула, начинающиеся на букву (исключения для генерации по букве)."""
    key = normalize_word(letter)
    if not key:
        return []
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian FROM word_pool WHERE norwegian LIKE ? ORDER BY RANDOM() LIMIT ?",
            (key + "%", limit),
        ) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def set_pool_meta(pool_id: int, level: str = None, topics=None):
    """Записать уровень + теги-темы слова (мульти-тег). topics — список ключей."""
    db = await _conn()
    try:
        if level is not None:
            await db.execute("UPDATE word_pool SET level = ? WHERE id = ?", (level, pool_id))
        if topics is not None:
            await db.execute("DELETE FROM word_topics WHERE pool_id = ?", (pool_id,))
            rows = [(pool_id, t) for t in topics if t]
            if rows:
                await db.executemany("INSERT OR IGNORE INTO word_topics (pool_id, topic) VALUES (?, ?)", rows)
        await db.commit()
    finally:
        await _release(db)


async def pool_missing_description(limit: int = 1):
    """Слова без описания — для фоновой догенерации. Возвращает [{word, pos, translate}],
    чтобы описание генерилось под конкретное значение (часть речи + переводы)."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian, data FROM word_pool WHERE description IS NULL LIMIT ?", (limit,)) as cur:
            out = []
            for r in await cur.fetchall():
                d = json.loads(r["data"]) if r["data"] else {}
                out.append({"id": r["id"], "word": r["norwegian"], "pos": d.get("part_of_speech", ""), "translate": d.get("translate", {})})
            return out
    finally:
        await _release(db)


async def clear_all_descriptions():
    """Сбросить ВСЕ описания (description=NULL) — фон догенерит заново под новый промпт."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET description = NULL")
        await db.commit()
    finally:
        await _release(db)


async def pool_missing_meta(limit: int = 50):
    """Слова без уровня (level IS NULL) + их переводы — для пакетной классификации."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data FROM word_pool WHERE level IS NULL LIMIT ?", (limit,)
        ) as cur:
            out = []
            for r in await cur.fetchall():
                d = json.loads(r["data"]) if r["data"] else {}
                out.append({"id": r["id"], "word": r["norwegian"], "pos": d.get("part_of_speech", ""),
                            "translate": d.get("translate", {})})
            return out
    finally:
        await _release(db)


async def get_pool_stats():
    """Сводка по пулу: всего и сколько с эмбеддингом/озвучкой/описанием/уровнем.
    ДВА прохода по таблице вместо ~13 отдельных COUNT: строки жирные (tts/embedding
    BLOBы, ~334МБ), каждый лишний скан — это чтение всей таблицы с сетевого диска."""
    db = await _conn()
    try:
        async with db.execute(
            f"SELECT COUNT(*) AS t, SUM(embedding IS NOT NULL) AS e, SUM({has_tts_expr()}) AS a, "
            "SUM(description IS NOT NULL) AS d, SUM(level IS NOT NULL) AS c FROM word_pool") as cur:
            r = await cur.fetchone()
            total, emb, tts_n, descr, classified = (r[k] or 0 for k in ("t", "e", "a", "d", "c"))
        forms_by_pos = {p: {"with": 0, "total": 0} for p in FORMABLE_POS}
        forms_sum = formable_sum = noun_no_gender = 0
        countability = {"total": 0, "marked": 0, "uncountable": 0}
        async with db.execute(
            f"SELECT {_POS_COL} AS p, COUNT(*) AS t, SUM(forms IS NOT NULL) AS w, "
            "SUM(CASE WHEN forms IS NOT NULL AND COALESCE(json_extract(forms,'$.gender'),'')='' "
            "AND COALESCE(json_extract(forms,'$.uninflectable'),0)=0 "   # несклоняемые (fjor, fru…) — не «застрявшие»
            "THEN 1 ELSE 0 END) AS ng, "
            # исчисляемость нунов — в том же проходе (отдельный LIKE-скан стоил ~4с)
            "SUM(json_extract(forms,'$.uncountable') IS NOT NULL) AS cm, "
            "SUM(CASE WHEN json_extract(forms,'$.uncountable') THEN 1 ELSE 0 END) AS cu "
            "FROM word_pool GROUP BY p") as cur:
            async for r in cur:
                if r["p"] in forms_by_pos:
                    forms_by_pos[r["p"]] = {"with": r["w"] or 0, "total": r["t"] or 0}
                    forms_sum += r["w"] or 0
                    formable_sum += r["t"] or 0
                if r["p"] == "noun":
                    noun_no_gender = r["ng"] or 0
                    countability = {"total": r["t"] or 0, "marked": r["cm"] or 0,
                                    "uncountable": r["cu"] or 0}
        return {
            "total": total,
            "embedding": emb,
            "tts": tts_n,
            "description": descr,
            "classified": classified,
            # формы считаем только у formable-частей речи (иначе числитель > знаменателя)
            "forms": forms_sum,
            "formable": formable_sum,
            "forms_by_pos": forms_by_pos,
            # «застрявшие»: сущ. с формами, но БЕЗ рода (нет артикля en/ei/et) — анти-залип/качество
            "noun_no_gender": noun_no_gender,
            "countability": countability,
        }
    finally:
        await _release(db)


async def get_pool_topics_counts():
    """[{topic, count}] по непустым темам (для фильтра)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT topic, COUNT(*) c FROM word_topics GROUP BY topic ORDER BY c DESC"
        ) as cur:
            return [{"topic": r["topic"], "count": r["c"]} for r in await cur.fetchall()]
    finally:
        await _release(db)


def resolve_compound(norwegian: str, data: dict = None):
    """ЕДИНСТВЕННАЯ точка правды по разбору составного слова. Порядок источников:
      1. ordbank leddanalyse (авторитет, ~106k) — для слов, которые банк знает;
      2. data.compound — LLM-разбор (ручной из карточки / генерация) для нюордов ВНЕ банка.
    Форма ответа одна и та же ({forledd, fuge, etterledd, marked, parts}), поэтому вызывающему
    всё равно, откуда пришёл разбор. Раньше фолбэк на data.compound знал ТОЛЬКО get_pool_meta,
    а флешкарта и compound_index_loop звали ordbank.compound() напрямую — и не видели
    LLM-разобранные слова. Все читатели обязаны идти через этот резолвер.
    Банк НЕ копируем в users.db: ordbank.db пересобирается целиком (ordbank.yml + bank_repair),
    копия стала бы протухающим зеркалом с обязанностью ре-импорта."""
    from db import ordbank
    c = ordbank.compound(norwegian)
    if c:
        return c
    lc = (data or {}).get("compound")
    if isinstance(lc, dict) and lc.get("forledd") and lc.get("etterledd"):
        fl, el = lc["forledd"], lc["etterledd"]
        return {"forledd": fl, "fuge": lc.get("fuge") or "", "etterledd": el,
                "marked": None, "parts": [fl, el]}
    return None


_COMPOUND_MAX_DEPTH = 5   # уровней разложения: реальные композиты редко глубже 3, 5 — с запасом


async def _pool_data_of(db, norwegian: str):
    """data слова из пула по лемме (для рекурсии дерева). Нет в пуле → None (разбор всё равно
    возьмётся из банка, просто без перевода). ORDER BY id — тот же выбор омонима, что у
    get_pool_id/get_pool_meta (иначе перевод узла берётся из другой записи омонима)."""
    async with db.execute("SELECT data FROM word_pool WHERE norwegian = ? ORDER BY id LIMIT 1", (norwegian,)) as cur:
        r = await cur.fetchone()
    if not r or not r["data"]:
        return None
    try:
        return json.loads(r["data"])
    except Exception:
        return None


def _node_tr(data, lang: str):
    """До 2 переводов узла на языке интерфейса (фолбэк ru→en). Нет в пуле → []."""
    tr = (data or {}).get("translate") or {}
    arr = tr.get(lang) or tr.get("ru") or tr.get("en") or []
    return [t for t in arr[:2] if t]


async def compound_tree(db, norwegian: str, data: dict = None, lang: str = "ru",
                        depth: int = _COMPOUND_MAX_DEPTH, seen=frozenset()):
    """Рекурсивное дерево разбора составного слова: узел = {word, tr, fuge, children}.
    children непусты ТОЛЬКО если слово составное (ветвление бинарное: forledd/etterledd),
    fuge узла — соединитель между его детьми. Разбор каждого узла — через единый
    resolve_compound (банк → LLM-фолбэк), поэтому вручную разобранные подслова тоже ветвятся.
    Защита: лимит глубины + seen (часть == предок / взаимная ссылка в банке → не зацикливаемся)."""
    node = {"word": norwegian, "tr": _node_tr(data, lang), "fuge": "", "children": []}
    if depth <= 0:
        return node
    c = resolve_compound(norwegian, data)
    if not c:
        return node
    node["fuge"] = c["fuge"]
    seen2 = seen | {norwegian}
    for part in (c["forledd"], c["etterledd"]):
        if not part or part in seen2:      # цикл — обрываем ветку листом, без рекурсии
            node["children"].append({"word": part, "tr": [], "fuge": "", "children": []})
            continue
        pdata = await _pool_data_of(db, part)
        node["children"].append(await compound_tree(db, part, pdata, lang, depth - 1, seen2))
    return node


async def get_pool_meta(word: str, user_id: int = None, pool_id: int = None, lang: str = "ru"):
    """Темы и уровень слова из пула (для показа в карточке). None — нет в пуле.
    inLearning — есть ли слово в Учёбе пользователя (в любом его словаре).
    ОМОНИМЫ: пул хранит отдельную запись на (norwegian, pos), поэтому по одному norwegian их бывает
    несколько (напр. `ro` — сущ. и глаг.). Если передан pool_id (клик по конкретной строке выдачи) —
    берём ИМЕННО ту запись; иначе первую по norwegian (навигация по лемме: синонимы/части композита)."""
    key = normalize_word(word)
    if not key:
        return None
    db = await _conn()
    try:
        row = None
        # видимость модерации применяется ТОЛЬКО при переданном user_id (запрос пользователя):
        # неодобренное (approved=0) слово видит лишь автор — иначе перебором pool_id читались чужие
        # pending. Без user_id (внутренние вызовы) фильтра нет («? IS NULL» → условие всегда истинно).
        vis = "AND (COALESCE(approved,1) = 1 OR created_by = ? OR ? IS NULL)"
        if pool_id:
            async with db.execute(f"SELECT id, level, data, forms, freq, {has_tts_expr()} AS has_tts "
                                  f"FROM word_pool WHERE id = ? {vis}", (pool_id, user_id, user_id)) as cur:
                row = await cur.fetchone()
        if row is None:
            # ORDER BY id — ОБЯЗАТЕЛЕН: без него SQLite шёл по уникальному индексу (norwegian, pos)
            # и выбирал омонима ПО АЛФАВИТУ части речи (mot: 'noun' < 'preposition' → «мужество»),
            # тогда как описание/синонимы берут запись через get_pool_id (ORDER BY id → предлог).
            # Карточка склеивала описание одного омонима с переводом/POS другого (120 слов из 274).
            async with db.execute(f"SELECT id, level, data, forms, freq, {has_tts_expr()} AS has_tts "
                                  f"FROM word_pool WHERE norwegian = ? {vis} ORDER BY id LIMIT 1",
                                  (key, user_id, user_id)) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        async with db.execute("SELECT topic FROM word_topics WHERE pool_id = ?", (row["id"],)) as cur:
            topics = [r["topic"] for r in await cur.fetchall()]
        in_learning = False
        if user_id:
            async with db.execute(
                "SELECT 1 FROM dict_words WHERE pool_id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?) LIMIT 1",
                (row["id"], user_id)) as cur:
                in_learning = (await cur.fetchone()) is not None
        d = json.loads(row["data"]) if row["data"] else {}
        compound = resolve_compound(key, d)   # банк → LLM-фолбэк (единый резолвер)
        if compound:
            # + рекурсивное дерево частей: подслово, которое само составное, ветвится дальше
            # (карточка показывает это в попапе части). Плоские поля не трогаем — их читают
            # cwSegs на фронте и флешкарта.
            compound = {**compound, "children": (await compound_tree(db, key, d, lang))["children"]}
        # формы: сущ./глаг./прил. — из word_pool.forms (ordbank); местоимения/притяжательные форм в
        # колонке не имеют → берём курируемую парадигму (obj для личных, neuter/plural для притяж.),
        # чтобы карточка показывала формы ВСЕХ частей речи, у которых они есть в системе.
        forms = json.loads(row["forms"]) if row["forms"] else None
        if not forms:
            from db.learning import PRONOUN_PARADIGM
            para = PRONOUN_PARADIGM.get(key)
            if para:
                forms = {"pos": "pronoun", **para}
        return {
            "level": row["level"], "topics": topics, "pool_id": row["id"],
            "part_of_speech": d.get("part_of_speech", ""),
            "translate": d.get("translate", {}),
            "forms": forms,
            "compound": compound,
            # проверено ли слово на составность (банк-разбор ИЛИ ручной LLM): фронт по этому
            # прячет пункт меню «Разобрать состав» у уже разобранных/простых слов
            "compoundChecked": bool(compound) or bool(d.get("compound_checked")),
            "hasTts": bool(row["has_tts"]),
            "freq": row["freq"], "freqBand": freq_band(row["freq"]),
            "inLearning": in_learning,
        }
    finally:
        await _release(db)


async def get_pool_facets(q: str = None, topics=None, level: str = None, lang: str = None, user_id: int = None):
    """Динамические счётчики фильтров под текущий выбор (дизъюнктивный facet — каждая группа
    считается БЕЗ учёта собственного выбора, т.к. мультивыбор внутри группы = ИЛИ).
    Темы: число слов по каждой теме под (поиск + уровень), без учёта выбранных тем —
    иначе невыбранный чип показывал бы пересечение с уже выбранными (прыгало бы при клике).
    Уровни: распределение под (поиск + выбранные темы), без учёта самого выбранного уровня."""
    key = normalize_word(q) if q else None

    def base(use_level, use_topics):
        conds, params = [], []
        if user_id:
            conds.append("(COALESCE(approved,1) = 1 OR created_by = ?)")
            params.append(user_id)
        else:
            conds.append("COALESCE(approved,1) = 1")
        if key:
            c, p = _key_cond_with_forms(key, lang)
            conds.append(c)
            params += p
        if use_level and level:
            conds.append("level = ?")
            params.append(level)
        if use_topics and topics:
            marks = ",".join("?" for _ in topics)
            conds.append(f"id IN (SELECT pool_id FROM word_topics WHERE topic IN ({marks}))")
            params += list(topics)
        return conds, params

    db = await _conn()
    try:
        # темы — без учёта выбранных тем (только поиск + уровень)
        c1, p1 = base(use_level=True, use_topics=False)
        w1 = ("WHERE " + " AND ".join(c1)) if c1 else ""
        async with db.execute(
            f"SELECT topic, COUNT(DISTINCT pool_id) c FROM word_topics "
            f"WHERE pool_id IN (SELECT id FROM word_pool {w1}) GROUP BY topic ORDER BY c DESC", p1
        ) as cur:
            topics_out = [{"topic": r["topic"], "count": r["c"]} for r in await cur.fetchall()]

        # уровни — без учёта самого выбранного уровня (но с учётом выбранных тем)
        c2, p2 = base(use_level=False, use_topics=True)
        c2.append("level IS NOT NULL")
        w2 = "WHERE " + " AND ".join(c2)
        async with db.execute(
            f"SELECT level, COUNT(*) c FROM word_pool {w2} GROUP BY level", p2
        ) as cur:
            levels_out = [{"level": r["level"], "count": r["c"]} for r in await cur.fetchall()]
        return {"topics": topics_out, "levels": levels_out}
    finally:
        await _release(db)


async def get_pool_level_counts():
    """[{level, count}] по проставленным уровням."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT level, COUNT(*) c FROM word_pool WHERE level IS NOT NULL GROUP BY level"
        ) as cur:
            return [{"level": r["level"], "count": r["c"]} for r in await cur.fetchall()]
    finally:
        await _release(db)


# Кеш кандидатов для дистракторов/похожих слов: весь пул целиком, поэтому
# (а) парсим JSON в отдельном потоке — не блокируем event-loop на горячем пути,
# (б) держим короткий TTL — повторные запросы (несколько слов подряд) не перегружают БД и CPU.
_CAND_TTL = float(os.getenv("POOL_CAND_TTL_SEC", "30"))
_cand_cache = {"rows": None, "ts": 0.0}


def _invalidate_candidates():
    _cand_cache["rows"] = None


def _build_candidates(rows):
    return [{
        "id": r["id"],
        "norwegian": r["norwegian"],
        "data": json.loads(r["data"]) if r["data"] else {},
        "embedding": r["embedding"],  # сырые байты (декодирует llm)
    } for r in rows]


async def get_pool_candidates():
    """Все слова пула (id, norwegian, data, embedding) — для подбора дистракторов.
    Кешируется на _CAND_TTL; разбор JSON по всему пулу вынесен в поток (не блокирует loop)."""
    now = time.monotonic()
    cached = _cand_cache["rows"]
    if cached is not None and (now - _cand_cache["ts"]) < _CAND_TTL:
        return cached
    db = await _conn()
    try:
        # только общая база (approved=1): неодобренные юзер-слова НЕ должны стать дистракторами/синонимами
        async with db.execute(
            "SELECT id, norwegian, data, embedding FROM word_pool WHERE COALESCE(approved,1) = 1") as cur:
            rows = await cur.fetchall()
    finally:
        await _release(db)
    out = await asyncio.to_thread(_build_candidates, rows)
    _cand_cache["rows"] = out
    _cand_cache["ts"] = now
    return out


# --- Поддержка резидентного кеша эмбеддингов (embcache): дистракторы без живого KNN ---
async def get_pool_words_by_ids(ids):
    """{id: {norwegian, data(dict)}} по списку id — батч-загрузка слов-соседей для дистракторов."""
    if not ids:
        return {}
    marks = ",".join("?" for _ in ids)
    db = await _conn()
    try:
        async with db.execute(f"SELECT id, norwegian, data FROM word_pool WHERE id IN ({marks})", list(ids)) as cur:
            return {r["id"]: {"norwegian": r["norwegian"],
                              "data": json.loads(r["data"]) if r["data"] else {}}
                    for r in await cur.fetchall()}
    finally:
        await _release(db)


async def load_pool_embeddings():
    """(ids[list[int]], raw_embeddings[list[bytes]]) всех слов с эмбеддингом — вход для embcache.
    Только вычитанные (COALESCE(approved,1)=1): неодобренные юзер-слова не должны стать дистракторами
    (база — approved NULL/1)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, embedding FROM word_pool "
            "WHERE embedding IS NOT NULL AND COALESCE(approved, 1) = 1") as cur:
            ids, embs = [], []
            for r in await cur.fetchall():
                ids.append(r["id"]); embs.append(r["embedding"])
            return ids, embs
    finally:
        await _release(db)


async def set_pool_description(pool_id: int, description: dict):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET description = ? WHERE id = ?", (json.dumps(description, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_forms(pool_id: int, forms: dict):
    """Грамматические формы слова (JSON: {pos, ...}). forms IS NOT NULL = «обработано»."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?",
                         (json.dumps(forms, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_compound(pool_id: int, compound: dict = None):
    """Записать (или пометить отсутствие) разбор составного слова в data.compound — для слов,
    которых нет в ordbank leddanalyse (ручной LLM-разбор из карточки). compound=None → слово
    проверено и НЕ составное (флаг compound_checked, чтобы пункт меню исчез и не жёг квоту).
    Разбор кладём и в обратный индекс частей (word_pool_compounds) — как делает compound_index_loop,
    иначе разблокировка композитов по выученным основам это слово не увидит (курсор мог его пройти)."""
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian, data FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return
        data = json.loads(row["data"]) if row["data"] else {}
        if compound and compound.get("forledd") and compound.get("etterledd"):
            data["compound"] = {"forledd": compound["forledd"], "fuge": compound.get("fuge") or "",
                                "etterledd": compound["etterledd"]}
        else:
            data.pop("compound", None)
            compound = None
        data["compound_checked"] = True
        await db.execute("UPDATE word_pool SET data = ? WHERE id = ?",
                         (json.dumps(data, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await _release(db)
    if compound:
        try:
            from .compound_index import set_pool_compounds   # upsert + invalidate внутри
            await set_pool_compounds([(pool_id, row["norwegian"], compound["forledd"], compound["etterledd"])])
        except Exception:
            pass


async def clear_nonformable_forms():
    """Удаляет грамм. формы у слов, чья часть речи их не предполагает (наречия/предлоги/…).
    Чистит «мусор» от прошлой коллизии (местоимения как сущ., наречия как глаг.). Возвращает кол-во."""
    db = await _conn()
    try:
        cur = await db.execute(f"UPDATE word_pool SET forms = NULL WHERE forms IS NOT NULL AND NOT {_FORMABLE_SQL}")
        await db.commit()
        return cur.rowcount or 0
    finally:
        await _release(db)


async def clear_all_forms():
    """Занулить forms у ВСЕХ formable-слов (noun/verb/adjective) — разовый сброс для перегенерации
    под новые/уточнённые промпты. Non-formable не трогаем (их чистит clear_nonformable_forms).
    Фон (forms_loop через pos_missing_forms) перезаполнит. Возвращает кол-во занулённых."""
    db = await _conn()
    try:
        cur = await db.execute(f"UPDATE word_pool SET forms = NULL WHERE forms IS NOT NULL AND {_FORMABLE_SQL}")
        await db.commit()
        return cur.rowcount or 0
    finally:
        await _release(db)


async def set_pool_pos(pool_id: int, pos: str):
    """Проставить часть речи в data.part_of_speech И в колонке pos (каноничный POS — колонка
    участвует в ключе (norwegian,pos) и в дедупе get_or_create_pool, иначе рассинхрон → дубли).
    Если каноничная запись (norwegian, pos) уже существует — текущую (с пустым/иным pos) СЛИВАЕМ в неё."""
    pos = normalize_pos(pos)
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian, COALESCE(pos,'') p FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return
        if row["p"] == pos:                              # колонка уже каноничная — синхроним только data
            await db.execute("UPDATE word_pool SET data = json_set(data, '$.part_of_speech', ?) WHERE id = ?", (pos, pool_id))
            await db.commit()
            return
        no = row["norwegian"]
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ? AND id <> ?",
                              (no, pos, pool_id)) as cur:
            other = await cur.fetchone()
    finally:
        await _release(db)
    if other:                                            # каноничная запись уже есть → слить дубль в неё
        from .pool_dedup import merge_pool_words
        await merge_pool_words(other["id"], pool_id)
        return
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET data = json_set(data, '$.part_of_speech', ?), pos = ? WHERE id = ?",
                         (pos, pos, pool_id))
        await db.commit()
    finally:
        await _release(db)


async def pos_uncategorized(limit: int = 20):
    """[(id, norwegian, data_dict)] — слова с пустой/неканоничной частью речи (колонка pos NOT IN
    POS_KEYS) — «прочее», кандидаты на переразметку (pos_loop)."""
    marks = ",".join("?" for _ in POS_KEYS)
    db = await _conn()
    try:
        async with db.execute(
            f"SELECT id, norwegian, data FROM word_pool WHERE {_POS_COL} NOT IN ({marks}) ORDER BY id LIMIT ?",
            (*[k.lower() for k in POS_KEYS], limit),
        ) as cur:
            return [(r["id"], r["norwegian"], json.loads(r["data"]) if r["data"] else {})
                    for r in await cur.fetchall()]
    finally:
        await _release(db)


async def nouns_missing_countability(limit: int = 50):
    """[(id, norwegian)] — нуны с заполненными формами, но БЕЗ отметки исчисляемости
    (бэкфилл одного прохода: countability_loop; новые получают её сразу в forms_batch)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian FROM word_pool WHERE forms LIKE '%\"pos\": \"noun\"%' "
            "AND forms NOT LIKE '%uncountable%' ORDER BY id LIMIT ?",
            (limit,),
        ) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def countability_progress():
    """Прогресс разметки исчисляемости нунов: {total, marked, uncountable} (для админки)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(*) AS t, "
            "SUM(CASE WHEN forms LIKE '%uncountable%' THEN 1 ELSE 0 END) AS m, "
            "SUM(CASE WHEN forms LIKE '%\"uncountable\": true%' THEN 1 ELSE 0 END) AS u "
            "FROM word_pool WHERE forms LIKE '%\"pos\": \"noun\"%'") as cur:
            r = await cur.fetchone()
        return {"total": r["t"] or 0, "marked": r["m"] or 0, "uncountable": r["u"] or 0}
    finally:
        await _release(db)


async def merge_pool_forms(pool_id: int, patch: dict):
    """Дописать ключи в forms-JSON слова (не затирая остальные формы)."""
    db = await _conn()
    try:
        async with db.execute("SELECT forms FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            r = await cur.fetchone()
        try:
            forms = json.loads(r["forms"]) if r and r["forms"] else {}
        except Exception:
            forms = {}
        forms.update(patch)
        await db.execute("UPDATE word_pool SET forms = ? WHERE id = ?",
                         (json.dumps(forms, ensure_ascii=False), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def pos_missing_forms(category: str, limit: int = 20):
    """[(id, norwegian, data_dict)] — слова заданной части речи БЕЗ грамматических форм."""
    cond, params = _pos_cond(category)
    if not cond:
        return []
    db = await _conn()
    try:
        async with db.execute(
            f"SELECT id, norwegian, data FROM word_pool WHERE forms IS NULL AND {cond} ORDER BY id LIMIT ?",
            (*params, limit),
        ) as cur:
            return [(r["id"], r["norwegian"], json.loads(r["data"]) if r["data"] else {})
                    for r in await cur.fetchall()]
    finally:
        await _release(db)


async def delete_pool_word(norwegian: str):
    """Полностью удалить слово из общего пула (каскадом у всех) + почистить кэш запросов."""
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    pid = None
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            pid = r["id"] if r else None
        await db.execute("DELETE FROM word_pool WHERE norwegian = ?", (key,))
        await db.execute(
            "DELETE FROM query_cache WHERE query LIKE ? OR response LIKE ? OR response LIKE ?",
            (f"%{key}%", f"%{key}%", f"%{norwegian}%"),
        )
        await db.commit()
    finally:
        await _release(db)
    await vec_delete(pid)  # убрать из ANN-индекса
    _invalidate_candidates()  # слово больше не должно всплывать в «похожих»
    if pid:
        try:                          # убрать из резидентного кеша эмбеддингов (дистракторы)
            import embcache
            embcache.remove_vec(pid)
        except Exception:
            pass


# (основной_ключ, добор|None). Направление — к основному ключу, добор всегда ASC.
# Пара, а не строка с запятыми: основной ключ может сам содержать запятую (COALESCE(freq, -1)),
# и наивный split(",") ломал SQL → 500 на сортировке по частоте.
_POOL_SORTS = {
    "alpha": ("norwegian", None),
    "added": ("created_at", "id"),
    # уровень: A1<…<C2, непроставленные в конец; добор по алфавиту
    "level": ("CASE level WHEN 'A1' THEN 1 WHEN 'A2' THEN 2 WHEN 'B1' THEN 3 "
              "WHEN 'B2' THEN 4 WHEN 'C1' THEN 5 WHEN 'C2' THEN 6 ELSE 99 END", "norwegian"),
    # частотность: NULL → -1 (в хвост при DESC); добор по алфавиту; направление — кнопкой порядка
    "freq": ("COALESCE(freq, -1)", "norwegian"),
}


# Админ-фильтр «чего не хватает у слова» (значение → SQL-условие).
_MISSING_SQL = {
    "embedding": "embedding IS NULL",
    "description": "description IS NULL",
    "tts": no_tts_expr(),
    "meta": "level IS NULL",
    "forms": "forms IS NULL",
}

# Часть речи берём из КАНОНИЧЕСКОЙ колонки pos (нормализована к англ.: substantiv→noun…),
# точным равенством. Раньше фильтр шёл по json_extract(data.part_of_speech) через LIKE-подстроки,
# из-за чего '%noun%' цеплял 'pronoun', а '%verb%' — 'adverb' (костыли _POS_NOT). Канон + '=' это
# устраняет, и роутинг forms_loop по части речи становится точным.
_POS_COL = "lower(COALESCE(pos,''))"


def _pos_cond(category):
    """(sql, params) — точный фильтр по канон-части речи (колонка pos)."""
    if not category:
        return None, []
    return f"{_POS_COL} = ?", [category.lower()]


def _pos_inline(category):
    """Чистый SQL-фрагмент по части речи (константа, без плейсхолдеров) — для составных условий."""
    return f"({_POS_COL} = '{category.lower()}')"


# Части речи, у которых вообще бывают грамматические формы (склонение/спряжение/степени).
# Остальные (наречия, предлоги, союзы, числит., междометия, фразы…) форм не получают —
# их не гоняем через Gemini и не считаем «без форм».
FORMABLE_POS = ("noun", "verb", "adjective")
_FORMABLE_SQL = "(" + " OR ".join(_pos_inline(c) for c in FORMABLE_POS) + ")"
# «Без форм» = только словам, которым формы положены, но их ещё нет.
_MISSING_SQL["forms"] = f"forms IS NULL AND {_FORMABLE_SQL}"


async def get_pool_list(limit: int = 60, offset: int = 0, q: str = None,
                        topics=None, level: str = None, sort: str = "alpha", order: str = "asc",
                        missing: str = None, pos: str = None, user_id: int = None, lang: str = None,
                        embed_fn=None):
    """Список слов общего пула: поиск по норвежскому + языку интерфейса, фильтры тема/уровень/
    часть речи, фильтр missing, сортировка и пагинация."""
    conds, params = [], []
    # видимость модерации: общая база (approved=1) + личное расширение автора (его pending/rejected)
    if user_id:
        conds.append("(COALESCE(approved,1) = 1 OR created_by = ?)")
        params.append(user_id)
    else:
        conds.append("COALESCE(approved,1) = 1")
    key = normalize_word(q) if q else None
    if key:
        # норвежский (+å/ø/æ) + перевод на язык интерфейса + сведение словоформы
        # к лемме банка (gikk → gå) — грид находит слово по любой его форме
        c, p = _key_cond_with_forms(key, lang)
        conds.append(c)
        params += p
    if topics:
        marks = ",".join("?" for _ in topics)
        conds.append(f"id IN (SELECT pool_id FROM word_topics WHERE topic IN ({marks}))")
        params += list(topics)
    if level:
        conds.append("level = ?")
        params.append(level)
    if missing in _MISSING_SQL:
        conds.append(_MISSING_SQL[missing])
    pos_sql, pos_params = _pos_cond(pos)
    if pos_sql:
        conds.append(pos_sql)
        params += pos_params
    where = ("WHERE " + " AND ".join(conds)) if conds else ""

    primary, tie = _POOL_SORTS.get(sort, _POOL_SORTS["alpha"])
    direction = "DESC" if str(order).lower() == "desc" else "ASC"
    # направление — к основному ключу; добор (tie) всегда по возрастанию
    order_sql = f"{primary} {direction}" + (f", {tie}" if tie else "")

    db = await _conn()
    try:
        async with db.execute(f"SELECT COUNT(*) c FROM word_pool {where}", params) as cur:
            total = (await cur.fetchone())["c"]
        if key and sort == "relevance":
            # релевантность: тянем все совпадения (с потолком), считаем score в питоне, сортируем,
            # пагинируем — точное/префиксное/перевод-целиком выше похожих и подстрок-в-середине.
            async with db.execute(
                f"SELECT id, norwegian, data, freq, level, forms, {has_tts_expr()} AS has_tts,"
                f"(embedding IS NOT NULL) AS has_emb, (description IS NOT NULL) AS has_desc "
                f"FROM word_pool {where} LIMIT 800", params,
            ) as cur:
                allm = await cur.fetchall()
            qf = _fold_no(key)
            ranked = sorted(
                allm,
                key=lambda r: (-_pool_relevance(r["norwegian"], _loads(r["data"]), key, qf, r["freq"], lang), r["norwegian"]),
            )
            rows = ranked[offset:offset + limit]
        else:
            async with db.execute(
                f"SELECT id, norwegian, data, level, forms, {has_tts_expr()} AS has_tts,"
                f"(embedding IS NOT NULL) AS has_emb, (description IS NOT NULL) AS has_desc "
                f"FROM word_pool {where} ORDER BY {order_sql} LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ) as cur:
                rows = await cur.fetchall()
        # fuzzy-fallback: подстрока ничего не нашла (опечатка) → ищем по неточному совпадению
        # норвежского И переводов (любой язык), чтобы «молоок» находил melk. Только для чистого
        # текстового запроса (без фильтров тема/уровень/missing/pos) — иначе результат сбивает с толку.
        if total == 0 and key and len(key) >= 3 and not topics and not level and missing not in _MISSING_SQL and not pos_sql:
            fids = await fuzzy_pool_ids(db, q, limit)
            if fids:
                marks = ",".join("?" for _ in fids)
                async with db.execute(
                    f"SELECT id, norwegian, data, level, forms, {has_tts_expr()} AS has_tts,"
                    f"(embedding IS NOT NULL) AS has_emb, (description IS NOT NULL) AS has_desc "
                    f"FROM word_pool WHERE id IN ({marks})", fids,
                ) as cur:
                    frows = await cur.fetchall()
                pos_order = {pid: n for n, pid in enumerate(fids)}  # сохранить порядок по fuzzy-скору
                rows = sorted(frows, key=lambda r: pos_order.get(r["id"], 1 << 30))
                total = len(rows)
        # семантический fallback: ни подстрока, ни fuzzy ничего не дали → ищем по смыслу
        # (эмбеддинг запроса → ближайшие по косинусу через vec_words). embed_fn инжектится из
        # роутера (квота-aware embed_text). Только для чистого текстового запроса.
        if (not rows) and embed_fn and key and len(key) >= 3 and not topics and not level and missing not in _MISSING_SQL and not pos_sql:
            qvec = None
            try:
                qvec = await embed_fn(q)
            except Exception:
                qvec = None
            if qvec:
                raw = np.asarray(qvec, dtype=np.float16).tobytes()
                near = await vec_nearest_rows(raw, 30) or []
                sids = [r["id"] for r in near if r.get("distance") is not None and r["distance"] <= _SEM_MAX_DIST][:_SEM_MAX_RESULTS]
                if sids:
                    marks = ",".join("?" for _ in sids)
                    vis, vparams = "", []
                    if user_id:
                        vis = " AND (COALESCE(approved,1) = 1 OR created_by = ?)"
                        vparams = [user_id]
                    async with db.execute(
                        f"SELECT id, norwegian, data, level, forms, {has_tts_expr()} AS has_tts,"
                        f"(embedding IS NOT NULL) AS has_emb, (description IS NOT NULL) AS has_desc "
                        f"FROM word_pool WHERE id IN ({marks}){vis}", (*sids, *vparams),
                    ) as cur:
                        srows = await cur.fetchall()
                    sem_order = {pid: n for n, pid in enumerate(sids)}
                    ranked = sorted(srows, key=lambda r: sem_order.get(r["id"], 1 << 30))
                    total = len(ranked)
                    rows = ranked[offset:offset + limit]
        # темы страницы — одним запросом (без N+1)
        ids = [r["id"] for r in rows]
        topic_map = {}
        if ids:
            marks = ",".join("?" for _ in ids)
            async with db.execute(
                f"SELECT pool_id, topic FROM word_topics WHERE pool_id IN ({marks})", ids
            ) as cur:
                for tr in await cur.fetchall():
                    topic_map.setdefault(tr["pool_id"], []).append(tr["topic"])
        # какие слова страницы уже в Учёбе пользователя (в любом его словаре) — одним запросом
        in_learning = set()
        if user_id and ids:
            marks = ",".join("?" for _ in ids)
            async with db.execute(
                f"SELECT DISTINCT pool_id FROM dict_words "
                f"WHERE pool_id IN ({marks}) AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)",
                (*ids, user_id),
            ) as cur:
                in_learning = {tr["pool_id"] for tr in await cur.fetchall()}
        words = []
        for r in rows:
            d = json.loads(r["data"]) if r["data"] else {}
            words.append({
                "word": r["norwegian"], "pool_id": r["id"], "translate": d.get("translate", {}),
                "part_of_speech": d.get("part_of_speech", ""),
                "level": r["level"], "topics": topic_map.get(r["id"], []),
                "hasTts": bool(r["has_tts"]),
                "hasEmbedding": bool(r["has_emb"]), "hasDescription": bool(r["has_desc"]),
                "forms": json.loads(r["forms"]) if r["forms"] else None,
                "inLearning": r["id"] in in_learning,
            })
        return {"total": total, "words": words}
    finally:
        await _release(db)


async def get_pool_ids(q: str = None, topics=None, level: str = None):
    """Все id слов пула, подходящих под фильтр (тот же WHERE, что и get_pool_list) — без пагинации."""
    conds, params = [], []
    key = normalize_word(q) if q else None
    if key:
        conds.append("(norwegian LIKE ? OR data LIKE ?)")
        params += [f"%{key}%", f"%{key}%"]
    if topics:
        marks = ",".join("?" for _ in topics)
        conds.append(f"id IN (SELECT pool_id FROM word_topics WHERE topic IN ({marks}))")
        params += list(topics)
    if level:
        conds.append("level = ?")
        params.append(level)
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    db = await _conn()
    try:
        async with db.execute(f"SELECT id FROM word_pool {where}", params) as cur:
            return [r["id"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def search_pool(prefix: str, limit: int = 10, lang: str = None):
    """Автокомплит по общему пулу + банку форм (решения юзера, 3.07):
    1) пул по норвежскому префиксу; 2) пул ПО ЛЮБОЙ СЛОВОФОРМЕ через ordbank
    (gikk → gå, с пометкой viaForm); 3) пул по переводам; 4) fuzzy-слой ВСЕГДА
    (опечатки: kjoken → kjøkken); 5) добор вне пула — леммы нашего банка (частотные
    сначала), а не только лексикон."""
    from db import ordbank
    key = normalize_word(prefix)
    if not key:
        return []
    limit = max(1, min(50, limit))   # защита: limit<1 → LIMIT -1 = без ограничения (весь пул), кап сверху
    db = await _conn()
    try:
        out, have = [], set()

        def _row(r, via=None):
            d = json.loads(r["data"]) if r["data"] else {}
            item = {"word": r["norwegian"], "translate": d.get("translate", {}),
                    "part_of_speech": d.get("part_of_speech", ""), "inPool": True}
            if via:
                item["viaForm"] = via
            return item

        # 1) пул: точное/префиксное совпадение норвежского + вхождение в переводы
        async with db.execute(
            "SELECT norwegian, data FROM word_pool "
            "WHERE norwegian LIKE ? OR data LIKE ? "
            "ORDER BY (CASE WHEN norwegian = ? THEN 0 WHEN norwegian LIKE ? THEN 1 ELSE 2 END), "
            "norwegian LIMIT ?",
            (key + "%", "%" + key + "%", key, key + "%", limit),
        ) as cur:
            for r in await cur.fetchall():
                have.add(r["norwegian"].lower())
                out.append(_row(r))

        # 2) словоформа → лемма пула (gikk → gå): точная форма, затем формы по префиксу
        cands = [(key, w, p) for w, p in ordbank.exact_form(key)]
        if len(cands) < 6:
            cands += [(f, w, p) for f, w, p in ordbank.prefix_forms(key, 10)]
        for form, lemma, pos in cands:
            if len(out) >= limit or lemma in have:
                continue
            async with db.execute(
                "SELECT norwegian, data FROM word_pool WHERE norwegian = ? AND pos = ?",
                (lemma, pos)) as cur:
                r = await cur.fetchone()
            if r:
                have.add(r["norwegian"].lower())
                out.append(_row(r, via=form))

        # 3) fuzzy-слой всегда (не только при пустой выдаче) — опечатки и раскладки
        if len(key) >= 3 and len(out) < limit:
            fids = await fuzzy_pool_ids(db, prefix, limit * 2)
            if fids:
                marks = ",".join("?" for _ in fids)
                async with db.execute(
                        f"SELECT id, norwegian, data FROM word_pool WHERE id IN ({marks})", fids) as curf:
                    byid = {r["id"]: r for r in await curf.fetchall()}
                for fid in fids:
                    if len(out) >= limit:
                        break
                    r = byid.get(fid)
                    if not r or r["norwegian"].lower() in have:
                        continue
                    have.add(r["norwegian"].lower())
                    out.append(_row(r))

        # 4) добор ВНЕ пула: леммы нашего банка (частотные сначала — zipf из лексикона),
        #    плюс формы банка (viaForm), плюс лексикон для слов вне банка
        if len(out) < limit:
            bank = [(w, p, None) for w, p in ordbank.prefix_lemmas(key, limit * 3)]
            bank += [(w, p, f) for f, w, p in ordbank.prefix_forms(key, limit)]
            seen_b, uniq = set(), []
            for w, p, via in bank:
                if w.lower() in have or (w, p) in seen_b:
                    continue
                seen_b.add((w, p))
                uniq.append((w, p, via))
            zipf = {}
            if uniq:
                marks = ",".join("?" for _ in uniq)
                try:
                    async with db.execute(
                            f"SELECT word, zipf FROM nb_lexicon WHERE word IN ({marks})",
                            [w for w, _, _ in uniq]) as cur2:
                        zipf = {r["word"]: r["zipf"] for r in await cur2.fetchall()}
                except Exception:
                    pass
            uniq.sort(key=lambda t: -(zipf.get(t[0]) or 0))
            for w, p, via in uniq:
                if len(out) >= limit:
                    break
                have.add(w.lower())
                item = {"word": w, "translate": {}, "part_of_speech": p, "inPool": False,
                        "freq": zipf.get(w), "freqBand": freq_band(zipf.get(w)) if zipf.get(w) else None}
                if via:
                    item["viaForm"] = via
                out.append(item)
        # 5) хвост из лексикона (слова вне банка: составные и пр.), частотные сначала
        if len(out) < limit:
            try:
                async with db.execute(
                    "SELECT word, zipf FROM nb_lexicon WHERE word >= ? AND word < ? "
                    "ORDER BY zipf DESC LIMIT ?",
                    (key, key + "\uffff", limit * 3),
                ) as cur2:
                    for r in await cur2.fetchall():
                        if r["word"].lower() in have:
                            continue
                        out.append({"word": r["word"], "translate": {}, "part_of_speech": "",
                                    "inPool": False, "freq": r["zipf"], "freqBand": freq_band(r["zipf"])})
                        if len(out) >= limit:
                            break
            except Exception:
                pass
        return out
    finally:
        await _release(db)


# Модерация и жалобы «не учить» вынесены в db/pool_moderation.py (реэкспорт ниже).
from .pool_moderation import (  # noqa: E402,F401
    pending_words, pending_count, set_word_approval,
    report_word, skip_word, reported_words, reported_count, resolve_report,
)
from .pool_freq import (  # noqa: E402,F401
    freq_band, pool_by_freq, pool_by_freq_topics, freq_pending, set_pool_freq, set_pool_freq_bulk,
)
from .pool_dedup import (  # noqa: E402,F401
    dedup_pending, mark_dedup, pool_usage_count, nearest_other, merge_pool_words, dedup_progress,
)

# Аксессоры фоновых очередей (TTS/эмбеддинги/переводы/ё) вынесены в pool_queues —
# реэкспорт держит db.__init__ и `from db.pool import …` без изменений.
from .pool_queues import (  # noqa: E402,F401
    get_pool_tts,
    set_pool_tts,
    set_pool_embedding,
    get_pool_embeddings_raw,
    get_pool_embeddings_page,
    pool_missing_embedding,
    pool_missing_tts,
    translate_pending,
    mark_translate_done,
    sem_embed_pending,
    mark_sem_embed,
    tr_tts_pending,
    mark_tr_tts_done,
    yo_pending,
    mark_yo_done,
)
