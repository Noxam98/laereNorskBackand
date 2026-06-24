import json
import os
import time
import asyncio
import fuzzy
import numpy as np
from rapidfuzz.process import cdist as rf_cdist
from rapidfuzz.distance import OSA
from .core import _conn, _release, _now, normalize_word, vec_upsert, vec_delete, vec_nearest_rows


def _fold_no(s):
    """Свернуть норвежские å/ø/æ → a/o/ae (для поиска без норвежской раскладки: «male»↔«måle»)."""
    return (s or "").replace("å", "a").replace("ø", "o").replace("æ", "ae")


# SQL-выражение, складывающее å/ø/æ в колонке norwegian (для LIKE без учёта норв. букв)
_SQL_FOLD_NO = "replace(replace(replace(norwegian,'å','a'),'ø','o'),'æ','ae')"

_POOL_LANGS = {"ru", "ukr", "en", "pl", "lt"}  # языки интерфейса = ключи translate в data
_SEM_MAX_DIST = float(os.getenv("POOL_SEM_MAX_DIST", "0.52"))   # порог косинус-дистанции для семантики
_SEM_MAX_RESULTS = int(os.getenv("POOL_SEM_MAX_RESULTS", "12"))  # семантика — подсказка, не полный список


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


async def get_or_create_pool(norwegian: str, data: dict, created_by: int = None, approved: int = 1):
    """Вернуть id записи пула для (норвежское слово + часть речи), создав её при необходимости.
    Запись определяется парой (norwegian, pos) — омонимы (føde «еда»/«рожать») = разные записи.
    created_by/approved задаются только для НОВОЙ записи (существующую не трогаем — иначе можно
    случайно «разодобрить» общее слово). approved=0 + created_by=user → личное расширение."""
    key = normalize_word(norwegian)
    if not key:
        return None
    pos = ((data or {}).get("part_of_speech") or "")
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ? AND COALESCE(pos,'') = ?", (key, pos)) as cur:
            row = await cur.fetchone()
            if row:
                return row["id"]
        # частотность проставляем СРАЗУ при создании — из корпус-лексикона (нет в нём → 0.0)
        cur = await db.execute(
            "INSERT INTO word_pool (norwegian, data, created_at, pos, created_by, approved, freq) "
            "VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT zipf FROM nb_lexicon WHERE word = ?), 0.0))",
            (key, json.dumps(data, ensure_ascii=False), _now(), pos, created_by, approved, key),
        )
        await db.commit()
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


async def get_pool_by_id(pool_id: int):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "data": json.loads(row["data"]),
                "description": json.loads(row["description"]) if row["description"] else None,
                "embedding": row["embedding"],  # сырые байты
                "forms": json.loads(row["forms"]) if row["forms"] else None,
            }
    finally:
        await _release(db)


async def get_pool_tts(norwegian: str):
    key = normalize_word(norwegian)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT tts FROM word_pool WHERE norwegian = ?", (key,)) as cur:
            r = await cur.fetchone()
            return r["tts"] if r and r["tts"] else None
    finally:
        await _release(db)


async def set_pool_tts(norwegian: str, data: bytes):
    key = normalize_word(norwegian)
    if not key:
        return
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET tts = ? WHERE norwegian = ?", (data, key))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_embedding(pool_id: int, data):
    """data — бинарное представление вектора (float16 bytes)."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET embedding = ? WHERE id = ?", (data, pool_id))
        await db.commit()
    finally:
        await _release(db)
    await vec_upsert(pool_id, data)  # держим ANN-индекс в синхроне


async def get_pool_embeddings_raw():
    """[(id, embedding_raw)] для миграции форматов."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, embedding FROM word_pool WHERE embedding IS NOT NULL") as cur:
            return [(r["id"], r["embedding"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def get_pool_embeddings_page(limit: int = 1000, offset: int = 0):
    """[[id, hex(embedding)]] постранично — админ-выгрузка векторов наружу (мало RAM)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, hex(embedding) AS h FROM word_pool WHERE embedding IS NOT NULL ORDER BY id LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            return [[r["id"], r["h"]] for r in await cur.fetchall()]
    finally:
        await _release(db)


def freq_band(z):
    """Zipf-частотность → ключ-градация для UI (подписи во фронте по языкам)."""
    if z is None:
        return None
    if z >= 6:
        return "very_common"
    if z >= 5:
        return "common"
    if z >= 4:
        return "frequent"
    if z >= 3:
        return "occasional"
    if z >= 2:
        return "rare"
    return "very_rare"


async def pool_by_freq(limit: int = 80, level: str = None):
    """Слова пула по убыванию частотности (самые употребимые сначала; freq IS NULL — в хвост).
    [{pool_id, norwegian, translate, part_of_speech, freq}]. Фильтр по уровню CEFR."""
    conds, params = ["data IS NOT NULL", "COALESCE(learn_excluded, 0) = 0"], []
    if level:
        conds.append("level = ?"); params.append(level)
    sql = (f"SELECT id, norwegian, data, freq FROM word_pool WHERE {' AND '.join(conds)} "
           "ORDER BY freq IS NULL, freq DESC LIMIT ?")
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
                    out.append({"pool_id": r["id"], "norwegian": r["norwegian"], "translate": tr,
                                "part_of_speech": d.get("part_of_speech", ""), "freq": r["freq"]})
            return out
    finally:
        await _release(db)


async def pool_by_freq_topics(limit: int, level, topics):
    """Как pool_by_freq, но только слова с любой из тем `topics` (для фокуса Учёбы). По частоте."""
    if not topics:
        return []
    conds, params = ["wp.data IS NOT NULL", "COALESCE(wp.learn_excluded, 0) = 0"], []
    if level:
        conds.append("wp.level = ?"); params.append(level)
    marks = ",".join("?" for _ in topics)
    sql = (f"SELECT DISTINCT wp.id, wp.norwegian, wp.data, wp.freq FROM word_pool wp "
           f"JOIN word_topics wt ON wt.pool_id = wp.id "
           f"WHERE {' AND '.join(conds)} AND wt.topic IN ({marks}) "
           f"ORDER BY wp.freq IS NULL, wp.freq DESC LIMIT ?")
    params += list(topics); params.append(limit)
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
                    out.append({"pool_id": r["id"], "norwegian": r["norwegian"], "translate": tr,
                                "part_of_speech": d.get("part_of_speech", ""), "freq": r["freq"]})
            return out
    finally:
        await _release(db)


async def freq_pending(limit: int = 200):
    """Слова без частотности (freq IS NULL) — для фонового добора по корпусу. [(id, norwegian)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian FROM word_pool WHERE freq IS NULL LIMIT ?", (limit,)) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def set_pool_freq(pool_id: int, freq: float):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET freq = ? WHERE id = ?", (float(freq), pool_id))
        await db.commit()
    finally:
        await _release(db)


async def set_pool_freq_bulk(pairs):
    """pairs: [(id, freq)] — пакетная простановка частотности."""
    if not pairs:
        return 0
    db = await _conn()
    try:
        await db.executemany("UPDATE word_pool SET freq = ? WHERE id = ?",
                             [(float(f), pid) for pid, f in pairs])
        await db.commit()
        return len(pairs)
    finally:
        await _release(db)


async def dedup_pending(limit: int = 1):
    """Слова в очереди фонового дедупа (dedup_done=0, есть эмбеддинг). Новые — первыми (id DESC).
    Возвращает [(id, norwegian, data, embedding_raw)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data, embedding FROM word_pool "
            "WHERE COALESCE(dedup_done, 0) = 0 AND embedding IS NOT NULL ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            return [(r["id"], r["norwegian"], r["data"], r["embedding"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_dedup(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET dedup_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def pool_usage_count(pool_id: int):
    """Сколько раз слово используется в словарях (популярность)."""
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM dict_words WHERE pool_id = ?", (pool_id,)) as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def nearest_other(pool_id: int, embedding_raw, k: int = 4):
    """Ближайшие соседи по vec-индексу, кроме самого слова: [(id, norwegian, data, distance)].
    distance — косинус-дистанция (cos = 1 - distance). [] если индекс недоступен/пуст."""
    import numpy as np
    from .core import _f16_to_f32_bytes
    q = _f16_to_f32_bytes(embedding_raw, np)
    if q is None:
        return []
    db = await _conn()
    try:
        try:
            async with db.execute(
                "SELECT rowid, distance FROM vec_words WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (q, k + 1),
            ) as cur:
                knn = await cur.fetchall()
        except Exception:
            return []
        out = []
        for r in knn:
            if r["rowid"] == pool_id:
                continue
            async with db.execute("SELECT norwegian, data FROM word_pool WHERE id = ?", (r["rowid"],)) as c2:
                w = await c2.fetchone()
            if w:
                out.append((r["rowid"], w["norwegian"], w["data"], r["distance"]))
        return out
    finally:
        await _release(db)


async def merge_pool_words(winner_id: int, loser_id: int):
    """Слить loser в winner: перепривязать dict_words/user_words на победителя (OR IGNORE при
    UNIQUE-конфликте), удалить остатки и сам loser из word_pool/word_topics/vec_words. True при успехе."""
    if winner_id == loser_id:
        return False
    db = await _conn()
    try:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.execute("UPDATE OR IGNORE dict_words SET pool_id = ? WHERE pool_id = ?", (winner_id, loser_id))
        await db.execute("UPDATE OR IGNORE user_words SET pool_id = ? WHERE pool_id = ?", (winner_id, loser_id))
        await db.execute("DELETE FROM dict_words  WHERE pool_id = ?", (loser_id,))
        await db.execute("DELETE FROM user_words  WHERE pool_id = ?", (loser_id,))
        await db.execute("DELETE FROM word_topics WHERE pool_id = ?", (loser_id,))
        try:
            await db.execute("DELETE FROM vec_words WHERE rowid = ?", (loser_id,))
        except Exception:
            pass
        await db.execute("DELETE FROM word_pool WHERE id = ?", (loser_id,))
        await db.commit()
        return True
    finally:
        await _release(db)


async def dedup_progress():
    """(проверено, всего) для отслеживания фонового дедупа."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COUNT(*) tot, SUM(CASE WHEN COALESCE(dedup_done,0)=1 THEN 1 ELSE 0 END) done "
            "FROM word_pool WHERE embedding IS NOT NULL"
        ) as cur:
            r = await cur.fetchone()
            return (r["done"] or 0, r["tot"] or 0)
    finally:
        await _release(db)


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


async def pool_missing_embedding(limit: int = 1):
    """[(id, norwegian)] — записи без вектора. id нужен, чтобы эмбеддинг записать ИМЕННО
    в эту запись: омонимы (один norwegian → несколько записей с разным pos) имеют каждый
    свой вектор, а get_pool_id без pos попал бы в старшую и NULL у нужной не очистился бы."""
    db = await _conn()
    try:
        async with db.execute("SELECT id, norwegian FROM word_pool WHERE embedding IS NULL LIMIT ?", (limit,)) as cur:
            return [(r["id"], r["norwegian"]) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def pool_missing_tts(limit: int = 1):
    db = await _conn()
    try:
        async with db.execute("SELECT norwegian FROM word_pool WHERE tts IS NULL LIMIT ?", (limit,)) as cur:
            return [r["norwegian"] for r in await cur.fetchall()]
    finally:
        await _release(db)


async def translate_pending(limit: int = 10):
    """Слова без отметки translate_done — кандидаты на догенерацию переводов.
    Возвращает [(id, norwegian, data_dict)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data FROM word_pool WHERE COALESCE(translate_done, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], r["norwegian"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_translate_done(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET translate_done = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def update_pool_word(old_norwegian: str, translate: dict):
    """Правка слова в ОБЩЕМ пуле (для всех): обновить переводы (любые из ru/ukr/en/pl/lt/no)
    и, если изменилось норвежское слово (translate['no']), — переименовать (norwegian-ключ +
    data.word/translate.no). Сбрасывает emb_sem (пере-эмбеддинг по смыслу) и tts_tr_done.
    Возвращает {ok, norwegian} либо {error: not_found|exists}."""
    key = normalize_word(old_norwegian)
    db = await _conn()
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
            "UPDATE word_pool SET data = ?, norwegian = ?, emb_sem = 0, tts_tr_done = 0, translate_done = 1 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), new_key, pid),
        )
        await db.commit()
        return {"ok": True, "norwegian": new_key}
    finally:
        await _release(db)


async def replace_pool_word(old_norwegian: str, new_norwegian: str, data: dict):
    """Заменить слово в пуле стандартизованными данными (исправленное норв. слово + переводы +
    часть речи) и СБРОСИТЬ производное (эмбеддинг/формы/озвучку/флаги) — фон пересоздаст их
    заново для нового написания. pool_id сохраняется (ссылки словарей не рвутся).
    Возвращает {ok, norwegian} либо {error: not_found|exists}."""
    old_key = normalize_word(old_norwegian)
    new_key = normalize_word(new_norwegian) or old_key
    db = await _conn()
    try:
        async with db.execute("SELECT id FROM word_pool WHERE norwegian = ?", (old_key,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "not_found"}
        pid = row["id"]
        if new_key != old_key:
            async with db.execute("SELECT 1 FROM word_pool WHERE norwegian = ? AND id != ?", (new_key, pid)) as c2:
                if await c2.fetchone():
                    return {"error": "exists"}
        await db.execute(
            "UPDATE word_pool SET data = ?, norwegian = ?, embedding = NULL, emb_sem = 0, "
            "forms = NULL, tts = NULL, tts_tr_done = 0, translate_done = 0 WHERE id = ?",
            (json.dumps(data, ensure_ascii=False), new_key, pid),
        )
        await db.commit()
        return {"ok": True, "norwegian": new_key}
    finally:
        await _release(db)


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


async def sem_embed_pending(limit: int = 20):
    """Слова, у которых эмбеддинг ещё не пересчитан по смыслу (emb_sem = 0).
    Возвращает [(id, data_dict)]."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, data FROM word_pool WHERE COALESCE(emb_sem, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_sem_embed(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET emb_sem = 1 WHERE id = ?", (pool_id,))
        await db.commit()
    finally:
        await _release(db)


async def tr_tts_pending(limit: int = 5):
    """Слова, у которых озвучка переводов ещё не сгенерирована (tts_tr_done = 0).
    Возвращает [(id, data_dict)] — переводы берём из data.translate."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, data FROM word_pool WHERE COALESCE(tts_tr_done, 0) = 0 LIMIT ?", (limit,)
        ) as cur:
            return [(r["id"], json.loads(r["data"])) for r in await cur.fetchall()]
    finally:
        await _release(db)


async def mark_tr_tts_done(pool_id: int):
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET tts_tr_done = 1 WHERE id = ?", (pool_id,))
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
    """Сводка по пулу: всего и сколько с эмбеддингом/озвучкой/описанием/уровнем."""
    db = await _conn()
    try:
        async def one(sql):
            async with db.execute(sql) as cur:
                return (await cur.fetchone())[0]
        return {
            "total": await one("SELECT COUNT(*) FROM word_pool"),
            "embedding": await one("SELECT COUNT(*) FROM word_pool WHERE embedding IS NOT NULL"),
            "tts": await one("SELECT COUNT(*) FROM word_pool WHERE tts IS NOT NULL"),
            "description": await one("SELECT COUNT(*) FROM word_pool WHERE description IS NOT NULL"),
            "classified": await one("SELECT COUNT(*) FROM word_pool WHERE level IS NOT NULL"),
            # формы считаем только у formable-частей речи (иначе числитель > знаменателя)
            "forms": await one(f"SELECT COUNT(*) FROM word_pool WHERE forms IS NOT NULL AND {_FORMABLE_SQL}"),
            # сколько слов вообще должны иметь формы (сущ./глаг./прил.) — знаменатель для прогресса
            "formable": await one(f"SELECT COUNT(*) FROM word_pool WHERE {_FORMABLE_SQL}"),
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


async def get_pool_meta(word: str, user_id: int = None):
    """Темы и уровень слова из пула (для показа в карточке). None — нет в пуле.
    inLearning — есть ли слово в Учёбе пользователя (в любом его словаре)."""
    key = normalize_word(word)
    if not key:
        return None
    db = await _conn()
    try:
        async with db.execute("SELECT id, level, data, forms, freq, (tts IS NOT NULL) AS has_tts FROM word_pool WHERE norwegian = ?", (key,)) as cur:
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
        return {
            "level": row["level"], "topics": topics, "pool_id": row["id"],
            "part_of_speech": d.get("part_of_speech", ""),
            "translate": d.get("translate", {}),
            "forms": json.loads(row["forms"]) if row["forms"] else None,
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
            c, p = _key_cond(key, lang)
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
        async with db.execute("SELECT id, norwegian, data, embedding FROM word_pool") as cur:
            rows = await cur.fetchall()
    finally:
        await _release(db)
    out = await asyncio.to_thread(_build_candidates, rows)
    _cand_cache["rows"] = out
    _cand_cache["ts"] = now
    return out


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


async def set_pool_pos(pool_id: int, pos: str):
    """Проставить/исправить часть речи в data.part_of_speech."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET data = json_set(data, '$.part_of_speech', ?) WHERE id = ?",
                         (pos, pool_id))
        await db.commit()
    finally:
        await _release(db)


async def pos_uncategorized(limit: int = 20):
    """[(id, norwegian, data_dict)] — слова, у которых part_of_speech пустой или НЕ из таксономии
    (т.е. показываются как «прочее»). Кандидаты на переразметку части речи."""
    all_pats = [p for pats in _POS_LIKE.values() for p in pats]
    not_cond = " AND ".join(
        "lower(COALESCE(json_extract(data,'$.part_of_speech'),'')) NOT LIKE ?" for _ in all_pats)
    db = await _conn()
    try:
        async with db.execute(
            f"SELECT id, norwegian, data FROM word_pool WHERE {not_cond} ORDER BY id LIMIT ?",
            (*all_pats, limit),
        ) as cur:
            return [(r["id"], r["norwegian"], json.loads(r["data"]) if r["data"] else {})
                    for r in await cur.fetchall()]
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
    "tts": "tts IS NULL",
    "meta": "level IS NULL",
    "forms": "forms IS NULL",
}

# Категория части речи → подстроки в data.part_of_speech (значения разнятся: noun/substantiv/...).
_POS_LIKE = {
    "noun": ["%noun%", "%substantiv%", "%сущ%"],
    "verb": ["%verb%", "%глаг%", "%дієсл%"],
    "adjective": ["%adjective%", "%adjektiv%", "%adj%", "%прил%", "%прикм%"],
    "adverb": ["%adverb%", "%нареч%", "%присл%"],
    "preposition": ["%preposition%", "%preposisjon%", "%предлог%", "%прийм%"],
    "conjunction": ["%conjunction%", "%konjunksjon%", "%союз%", "%сполуч%"],
    "pronoun": ["%pronoun%", "%pronomen%", "%местоим%", "%займен%"],
    "determiner": ["%determiner%", "%determinativ%", "%артикл%", "%определит%"],
    "numeral": ["%numeral%", "%tallord%", "%числит%", "%числ%"],
    "interjection": ["%interjection%", "%interjeksjon%", "%междомет%", "%виг%"],
    "phrase": ["%phrase%", "%uttrykk%", "%фраз%", "%выраж%"],
}


_POS_COL = "lower(COALESCE(json_extract(data,'$.part_of_speech'),''))"
# Исключения от коллизий подстрок: «pronoun» содержит «noun», «adverb» содержит «verb».
# Без них фильтр noun/verb захватывал бы местоимения/наречия (и Gemini получал бы их формы).
_POS_NOT = {"noun": ["%pronoun%", "%pronomen%"], "verb": ["%adverb%"]}


def _pos_cond(category):
    """(sql, params) для фильтра по части речи через json_extract(data.part_of_speech)."""
    likes = _POS_LIKE.get(category)
    if not likes:
        return None, []
    sub = " OR ".join(f"{_POS_COL} LIKE ?" for _ in likes)
    sql, params = f"({sub})", list(likes)
    for nx in _POS_NOT.get(category, []):
        sql += f" AND {_POS_COL} NOT LIKE ?"; params.append(nx)
    return sql, params


def _pos_inline(category):
    """Чистый SQL-фрагмент по части речи (константы, без плейсхолдеров) — для составных условий."""
    likes = _POS_LIKE.get(category) or []
    sql = "(" + " OR ".join(f"{_POS_COL} LIKE '{p}'" for p in likes) + ")"
    for nx in _POS_NOT.get(category, []):
        sql += f" AND {_POS_COL} NOT LIKE '{nx}'"
    return f"({sql})"


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
        c, p = _key_cond(key, lang)   # норвежский (+å/ø/æ) + перевод на язык интерфейса
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
                f"SELECT id, norwegian, data, freq, level, forms, (tts IS NOT NULL) AS has_tts, "
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
                f"SELECT id, norwegian, data, level, forms, (tts IS NOT NULL) AS has_tts, "
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
                    f"SELECT id, norwegian, data, level, forms, (tts IS NOT NULL) AS has_tts, "
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
                        f"SELECT id, norwegian, data, level, forms, (tts IS NOT NULL) AS has_tts, "
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
    """Автокомплит по общему пулу: по норвежскому слову И по переводам (любой язык).
    Норвежские совпадения по префиксу — выше. Неточный (fuzzy) добор — по норвежскому
    и по переводу на `lang` (язык интерфейса), если точных совпадений мало."""
    key = normalize_word(prefix)
    if not key:
        return []
    db = await _conn()
    try:
        async with db.execute(
            "SELECT norwegian, data FROM word_pool "
            "WHERE norwegian LIKE ? OR data LIKE ? "
            "ORDER BY (CASE WHEN norwegian LIKE ? THEN 0 ELSE 1 END), norwegian LIMIT ?",
            (key + "%", "%" + key + "%", key + "%", limit),
        ) as cur:
            rows = await cur.fetchall()
            out, have = [], set()
            for r in rows:
                d = json.loads(r["data"]) if r["data"] else {}
                have.add(r["norwegian"].lower())
                out.append({"word": r["norwegian"], "translate": d.get("translate", {}),
                            "part_of_speech": d.get("part_of_speech", ""), "inPool": True})
            # неточный (fuzzy) добор по опечаткам — через общий кэш-индекс (норвежский + переводы).
            if len(out) < limit and len(key) >= 3:
                fids = await fuzzy_pool_ids(db, prefix, limit * 2)
                if fids:
                    marks = ",".join("?" for _ in fids)
                    async with db.execute(f"SELECT id, norwegian, data FROM word_pool WHERE id IN ({marks})", fids) as curf:
                        byid = {r["id"]: r for r in await curf.fetchall()}
                    for fid in fids:
                        if len(out) >= limit:
                            break
                        r = byid.get(fid)
                        if not r or r["norwegian"].lower() in have:
                            continue
                        d = json.loads(r["data"]) if r["data"] else {}
                        out.append({"word": r["norwegian"], "translate": d.get("translate", {}),
                                    "part_of_speech": d.get("part_of_speech", ""), "inPool": True})
                        have.add(r["norwegian"].lower())
            # добор из полного лексикона (слова, которых ещё нет в пуле) — по префиксу, частотные сначала
            if len(out) < limit:
                try:
                    async with db.execute(
                        "SELECT word, zipf FROM nb_lexicon WHERE word >= ? AND word < ? "
                        "ORDER BY zipf DESC LIMIT ?",
                        (key, key + "￿", limit * 3),
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


# ---------------- Модерация пользовательских слов (личное расширение → общая база) ----------------
async def pending_words(limit: int = 300, offset: int = 0):
    """Слова на модерации (approved=0) по всем юзерам — для админа, с автором."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT wp.id, wp.norwegian, wp.data, wp.pos, wp.created_at, wp.created_by, "
            "u.username AS author "
            "FROM word_pool wp LEFT JOIN users u ON u.id = wp.created_by "
            "WHERE COALESCE(wp.approved, 1) = 0 "
            "ORDER BY wp.created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            d = _loads(r["data"])
            out.append({
                "pool_id": r["id"], "word": r["norwegian"],
                "part_of_speech": r["pos"] or d.get("part_of_speech", ""),
                "translate": d.get("translate", {}),
                "level": d.get("level"), "topics": d.get("topics", []),
                "author": r["author"], "author_id": r["created_by"], "created_at": r["created_at"],
            })
        return out
    finally:
        await _release(db)


async def pending_count():
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM word_pool WHERE COALESCE(approved, 1) = 0") as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def set_word_approval(pool_id: int, approved: int):
    """approved: 1 — одобрить (в общую базу), 2 — отклонить (остаётся приватным у автора)."""
    db = await _conn()
    try:
        await db.execute("UPDATE word_pool SET approved = ? WHERE id = ?", (int(approved), pool_id))
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "approved": int(approved)}
    finally:
        await _release(db)


# ---------------- Жалобы «не учить» (мусорные слова → модерация → убрать из учёбы / оставить) ----------------
async def report_word(pool_id: int, user_id: int):
    """Пользователь жалуется «не учить». Убираем слово из ЕГО учёбы и решаем судьбу жалобы:
      • learn_excluded=1     → уже убрано из учёбы для всех (status=excluded, тихо);
      • report_dismiss_left>0 → админ ранее решил «оставить» → гасим жалобу (−1, status=dismissed);
      • иначе                 → ставим в очередь админа (reported=1, report_count+1, status=queued)."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT COALESCE(learn_excluded,0) le, COALESCE(report_dismiss_left,0) dl "
            "FROM word_pool WHERE id = ?", (pool_id,)) as cur:
            row = await cur.fetchone()
        if not row:
            return {"error": "not_found"}
        # «Отправить на модерацию» = в персональную СВАЛКУ юзера: учить НЕ будет НАВСЕГДА, независимо
        # от решения модератора. Убираем из его словарей (dict_words) и прогресса (user_words —
        # удаляем, это мусор, не «выучено»), и заносим в user_word_skips, чтобы suggest_words больше
        # никогда не предлагал это слово ЭТОМУ юзеру (даже если модератор слово оставит).
        await db.execute(
            "DELETE FROM dict_words WHERE pool_id = ? AND dict_id IN (SELECT id FROM dictionaries WHERE user_id = ?)",
            (pool_id, user_id))
        await db.execute("DELETE FROM user_words WHERE user_id = ? AND pool_id = ?", (user_id, pool_id))
        await db.execute("INSERT OR IGNORE INTO user_word_skips (user_id, pool_id, created_at) VALUES (?,?,?)", (user_id, pool_id, _now()))
        if row["le"]:
            status = "excluded"
        elif row["dl"] > 0:
            await db.execute("UPDATE word_pool SET report_dismiss_left = report_dismiss_left - 1 WHERE id = ?", (pool_id,))
            status = "dismissed"
        else:
            await db.execute("UPDATE word_pool SET reported = 1, report_count = COALESCE(report_count,0) + 1 WHERE id = ?", (pool_id,))
            status = "queued"
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "status": status}
    finally:
        await _release(db)


async def reported_words(limit: int = 300, offset: int = 0):
    """Слова с активными жалобами «не учить» (reported=1) — для админа."""
    db = await _conn()
    try:
        async with db.execute(
            "SELECT id, norwegian, data, pos, level, COALESCE(report_count,0) rc "
            "FROM word_pool WHERE COALESCE(reported,0) = 1 "
            "ORDER BY rc DESC, id DESC LIMIT ? OFFSET ?", (limit, offset)) as cur:
            rows = await cur.fetchall()
        out = []
        for r in rows:
            d = _loads(r["data"])
            out.append({
                "pool_id": r["id"], "word": r["norwegian"],
                "part_of_speech": r["pos"] or d.get("part_of_speech", ""),
                "translate": d.get("translate", {}),
                "level": r["level"] or d.get("level"),
                "reports": r["rc"],
            })
        return out
    finally:
        await _release(db)


async def reported_count():
    db = await _conn()
    try:
        async with db.execute("SELECT COUNT(*) c FROM word_pool WHERE COALESCE(reported,0) = 1") as cur:
            return (await cur.fetchone())["c"]
    finally:
        await _release(db)


async def resolve_report(pool_id: int, action: str):
    """Вердикт админа по жалобе:
      • 'exclude' — убрать из учёбы (learn_excluded=1, снять жалобу);
      • 'keep'    — оставить слово, следующие 5 жалоб гасить автоматически (report_dismiss_left=5)."""
    db = await _conn()
    try:
        if action == "exclude":
            await db.execute(
                "UPDATE word_pool SET learn_excluded = 1, reported = 0, report_count = 0, report_dismiss_left = 0 WHERE id = ?",
                (pool_id,))
        elif action == "keep":
            await db.execute(
                "UPDATE word_pool SET reported = 0, report_count = 0, report_dismiss_left = 5 WHERE id = ?",
                (pool_id,))
        else:
            return {"error": "bad_action"}
        await db.commit()
        return {"ok": True, "pool_id": pool_id, "action": action}
    finally:
        await _release(db)
