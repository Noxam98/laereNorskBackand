"""Детерминированные грамматические формы из Norsk ordbank (CC-BY) — вместо LLM.

Файл ordbank.db (~19МБ, 138k парадигм сущ./глаг./прил.) собирается офлайн
(~/norsk-harvest, логика build_forms харвестера: prop-леммы исключены, артикль
согласован с def_sg) и лежит рядом с users.db; при отсутствии скачивается из
Tigris (ordbank/ordbank.db). Нет файла или слова — None, форму сгенерит LLM-фолбэк.
"""
import json
import os
import sqlite3
import threading

from config import logger

_DIR = os.path.dirname(os.getenv("DATABASE_PATH", "users.db")) or "."
PATH = os.path.join(_DIR, "ordbank.db")
_conn = None
_lock = threading.Lock()
_ensured = False


def lookup(norwegian: str, pos: str):
    """Формы слова из банка: {pos, gender, def_sg, …} или None. Потокобезопасно, <1мс."""
    global _conn
    key = (norwegian or "").strip().lower()
    if not key or not pos:
        return None
    with _lock:
        if _conn is None:
            if not os.path.exists(PATH):
                return None
            _conn = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True, check_same_thread=False)
        row = _conn.execute(
            "SELECT forms FROM forms WHERE norwegian = ? AND pos = ?", (key, pos)).fetchone()
    return json.loads(row[0]) if row else None


async def ensure_file():
    """Разово при старте воркера: если ordbank.db нет рядом с БД — забрать из Tigris."""
    global _ensured
    if _ensured:
        return os.path.exists(PATH)
    _ensured = True
    if os.path.exists(PATH):
        logger.info(f"ordbank: {PATH} на месте")
        return True
    try:
        import storage
        data = await storage.get_object("ordbank/ordbank.db")
        if data:
            with open(PATH, "wb") as f:
                f.write(data)
            logger.info(f"ordbank: скачан из Tigris ({len(data) // 1024 // 1024} МБ)")
            return True
        logger.warning("ordbank: файла нет ни локально, ни в Tigris — формы будет делать LLM")
    except Exception as e:
        logger.warning(f"ordbank: не скачался ({e}) — формы будет делать LLM")
    return False


# ── фолбэк: живой ordbøkene (ord.uib.no) для слов НЕ из дампа (nyord и т.п.) ──
# Ходит только фоновый forms_loop; результат кэшируется в users.db (ordbank_ext),
# так что каждое слово ездит в сеть максимум один раз. '{}' в кэше = «в словаре нет».
_UIB = "https://ord.uib.no"
_G = {"Masc": "en", "Fem": "ei", "Neuter": "et"}


def _parse_uib(article: dict, pos: str):
    """paradigm_info статьи → наш forms-словарь (или None)."""
    for lem in article.get("lemmas") or []:
        pis = lem.get("paradigm_info") or []
        if pos == "noun":
            # дублет рода → Fem приоритетнее (конвенция дампа: ei jente/jenta)
            for want in ("Fem", "Masc", "Neuter"):
                for pi in pis:
                    tags = pi.get("tags") or []
                    if "NOUN" not in tags or want not in tags:
                        continue
                    out = {"pos": "noun", "gender": _G[want]}
                    for inf in pi.get("inflection") or []:
                        t, w = set(inf.get("tags") or []), inf.get("word_form")
                        if not w:
                            continue
                        if t >= {"Sing", "Def"}:
                            out["def_sg"] = w
                        elif t >= {"Plur", "Ind"}:
                            out["indef_pl"] = w
                        elif t >= {"Plur", "Def"}:
                            out["def_pl"] = w
                    if len(out) > 2:
                        return out
        elif pos == "verb":
            for pi in pis:
                if "VERB" not in (pi.get("tags") or []):
                    continue
                out = {"pos": "verb"}
                for inf in pi.get("inflection") or []:
                    t, w = set(inf.get("tags") or []), inf.get("word_form")
                    if not w:
                        continue
                    if "Pres" in t and "Part" not in t:
                        out.setdefault("present", w)
                    elif "Past" in t and "Part" not in t:
                        out.setdefault("past", w)
                    elif {"PerfPart"} & t or {"Past", "Part"} <= t:
                        out.setdefault("perfect", f"har {w}")
                if len(out) > 1:
                    return out
        elif pos == "adjective":
            for pi in pis:
                if "ADJ" not in (pi.get("tags") or []):
                    continue
                out = {"pos": "adjective"}
                for inf in pi.get("inflection") or []:
                    t, w = set(inf.get("tags") or []), inf.get("word_form")
                    if not w:
                        continue
                    if "Pos" in t and "Neuter" in t:
                        out.setdefault("neuter", w)
                    elif "Pos" in t and "Plur" in t:
                        out.setdefault("plural", w)
                    elif "Cmp" in t:
                        out.setdefault("comparative", w)
                    elif "Sup" in t and "Ind" in t:
                        out.setdefault("superlative", w)
                if len(out) > 1:
                    return out
    return None


def _uib_fetch(norwegian: str, pos: str):
    """Синхронный поход в ord.uib.no (зовётся через to_thread)."""
    import urllib.request, urllib.parse
    q = urllib.parse.quote(norwegian)
    with urllib.request.urlopen(f"{_UIB}/api/articles?w={q}&dict=bm&scope=ei", timeout=10) as r:
        ids = (json.load(r).get("articles") or {}).get("bm") or []
    for aid in ids[:3]:
        with urllib.request.urlopen(f"{_UIB}/bm/article/{aid}.json", timeout=10) as r:
            forms = _parse_uib(json.load(r), pos)
        if forms:
            return forms
    return None


async def lookup_online(norwegian: str, pos: str):
    """Кэш ordbank_ext → ord.uib.no → кэш. None = нигде нет (дальше LLM)."""
    import asyncio
    from db.core import _conn, _release
    key = (norwegian or "").strip().lower()
    if not key or pos not in ("noun", "verb", "adjective"):
        return None
    db = await _conn()
    try:
        async with db.execute(
                "SELECT forms FROM ordbank_ext WHERE norwegian=? AND pos=?", (key, pos)) as cur:
            r = await cur.fetchone()
        if r is not None:
            return json.loads(r["forms"]) or None   # '{}' = известный промах
        try:
            forms = await asyncio.to_thread(_uib_fetch, key, pos)
        except Exception as e:
            logger.warning(f"ordbøkene '{key}': {e}")
            return None   # сетевую ошибку НЕ кэшируем — попробуем в другой раз
        await db.execute("INSERT OR REPLACE INTO ordbank_ext VALUES (?,?,?,datetime('now'))",
                         (key, pos, json.dumps(forms or {}, ensure_ascii=False)))
        await db.commit()
        if forms:
            logger.info(f"ordbøkene: {key}/{pos} → формы из живого словаря")
        return forms
    finally:
        await _release(db)


# ── поиск: обратный индекс словоформ и автокомплит по леммам банка ────────────
def _q(sql, args):
    """Сырой запрос к банку (None, если файла/таблицы нет — старый ordbank.db без formindex)."""
    global _conn
    with _lock:
        if _conn is None:
            if not os.path.exists(PATH):
                return []
            _conn = sqlite3.connect(f"file:{PATH}?mode=ro", uri=True, check_same_thread=False)
        try:
            return _conn.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []


def exact_form(form: str):
    """Точная словоформа → [(лемма, pos)]: gikk → [(gå, verb)]. Леммы-сами-себя не считаем."""
    key = (form or "").strip().lower()
    if not key:
        return []
    return [(w, p) for w, p in _q(
        "SELECT norwegian, pos FROM formindex WHERE form = ?", (key,)) if w != key]


def prefix_forms(prefix: str, limit: int = 10):
    """Формы по префиксу → [(форма, лемма, pos)] — авто̇комплит «gik → gikk (gå)»."""
    key = (prefix or "").strip().lower()
    if len(key) < 2:
        return []
    return _q("SELECT form, norwegian, pos FROM formindex WHERE form LIKE ? "
              "AND form != norwegian LIMIT ?", (key + "%", limit))


def prefix_lemmas(prefix: str, limit: int = 20):
    """Леммы банка по префиксу → [(лемма, pos)] — добор автокомплита вне пула."""
    key = (prefix or "").strip().lower()
    if len(key) < 2:
        return []
    return _q("SELECT norwegian, pos FROM forms WHERE norwegian LIKE ? LIMIT ?",
              (key + "%", limit))


def compound(norwegian: str):
    """Разбор составного слова (sammensetning) из банка (leddanalyse): {forledd, fuge,
    etterledd, marked, parts:[forledd, etterledd]} или None, если слово не составное.
    forledd/etterledd — самостоятельные леммы (кликаются/ищутся в пуле); разбор
    рекурсивный (голова/первый элемент могут сами быть составными). marked — поверхность
    со швом («barne-hage»). Старый ordbank.db без таблицы compounds → None (_q гасит)."""
    key = (norwegian or "").strip().lower()
    if not key:
        return None
    rows = _q("SELECT forledd, fuge, etterledd, marked FROM compounds WHERE norwegian = ?", (key,))
    if not rows:
        return None
    forledd, fuge, etterledd, marked = rows[0]
    # части — самостоятельные леммы, кликаются/ищутся в пуле (pool.norwegian всегда lower); старые
    # дампы клали forledd/etterledd без .lower() → рантайм-нормализация, чтобы часть матчилась.
    fl, el = (forledd or "").lower(), (etterledd or "").lower()
    return {"forledd": fl, "fuge": fuge or "", "etterledd": el,
            "marked": marked, "parts": [fl, el]}
