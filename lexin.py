"""Живой Lexin (editorportal.oslomet.no) — человеческие переводы ОДНОГО слова при
добавлении в пул (решение юзера 3.07: банк даёт формы, Lexin — переводы, LLM — фолбэк).

Порт боевой логики харвестера (~/norsk-harvest/harvest_lexin.py): ru-запрос — разведчик
(норвежская база статей у Lexin общая: нет леммы в ru-ответе — нет нигде), остальные
словари параллельно; английский — бонусом из ru-ответа (includeEngLang=1, строки B-*).
Статья выбирается по лемме и POS (чужой kat не берётся), переводы чистятся к формату
прода. Сетевые ошибки → {} (дальше LLM-фолбэк, слово не страдает).
"""
import asyncio
import http.client
import json
import re
import urllib.parse
from collections import defaultdict

from config import logger

HDRS = {"Origin": "https://lexin.oslomet.no", "Referer": "https://lexin.oslomet.no/",
        "User-Agent": "learnNorsk backend (single-word lookups)"}
DICTS = [("bokmål-russisk", "Ru", "ru"), ("bokmål-ukrainsk", "Ukr", "ukr"),
         ("bokmål-polsk", "Pol", "pl"), ("bokmål-litauisk", "Lit", "lt"),
         ("bokmål-arabisk", "Ara", "ar")]
KAT2POS = {"subst": "noun", "substantiv": "noun", "verb": "verb",
           "adj": "adjective", "adjektiv": "adjective", "adv": "adverb", "adverb": "adverb",
           "prep": "preposition", "preposisjon": "preposition",
           "konj": "conjunction", "konjunksjon": "conjunction", "subjunksjon": "conjunction",
           "interj": "interjection", "interjeksjon": "interjection",
           "pron": "pronoun", "pronomen": "pronoun", "det": "determiner", "determinativ": "determiner"}
FUNC_POS = {"adverb", "preposition", "conjunction", "pronoun", "determiner", "interjection"}
_QUOTES = re.compile(r'["«»„“”\'‘’]')
_UKR_INF = re.compile(r".+ти(ся)?$")


def norm_translations(items):
    """Lexin-строки → чистые варианты формата прода (сплит ;,| и видовых пар)."""
    out = []
    for t in items or []:
        for chunk in re.split(r"[;,|]", t):
            v = chunk.strip()
            if not v:
                continue
            if "/" in v:
                v = v.split("/")[0].strip()
            v = _QUOTES.sub("", re.sub(r"\s*\(.*?\)\s*", " ", v)).strip(" .").strip()
            toks = v.split()
            pieces = toks if (len(toks) == 2 and all(_UKR_INF.match(x) for x in toks)) else [v]
            for p in pieces:
                if p and p.lower() not in (x.lower() for x in out):
                    out.append(p)
    return out[:5]


def _fetch(word: str, dict_slug: str):
    q = urllib.parse.quote(word)
    s = urllib.parse.quote(dict_slug)
    path = (f"/api/v1/findwords?searchWord={q}&lang={s}&page=1"
            f"&selectLang={s}&includeEngLang=1")
    conn = http.client.HTTPSConnection("editorportal.oslomet.no", timeout=15)
    try:
        conn.request("GET", path, headers=HDRS)
        r = conn.getresponse()
        body = r.read()
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}")
        return json.loads(body)
    finally:
        conn.close()


def _parse(data, word, want_pos, prefix):
    """Переводы главной (самой богатой) статьи с нашей леммой и POS. None = мимо."""
    flat = [e for g in (data.get("result") or []) for e in g]
    by_id = defaultdict(list)
    for e in flat:
        by_id[e["id"]].append(e)
    cand = []
    for aid, rows in by_id.items():
        lems = [e for e in rows if e["type"].endswith("-lem") and e["type"][0] in "EN"]
        if any((e["text"] or "").strip().lower() == word for e in lems):
            cand.append(rows)
    if not cand:
        return None

    def kat_of(rows):
        for e in rows:
            if e["type"].endswith("-kat") and e["type"][0] in "EN":
                t = (e["text"] or "").strip().lower()
                return KAT2POS.get(t.split()[0] if t else "", None)
        return None

    match = [r for r in cand if kat_of(r) == want_pos]
    unknown = [] if want_pos in FUNC_POS else [r for r in cand if kat_of(r) is None]
    for rows in sorted(match, key=lambda r: -len(r)) + sorted(unknown, key=lambda r: -len(r)):
        keep = sorted((e for e in rows if e["type"].split("-", 1)[0] in ("E", "N", prefix)),
                      key=lambda e: e["sub_id"])
        tr = [(e["text"] or "").strip() for e in keep
              if e["type"].split("-", 1)[1] == "lem" and not (e["type"][0] in "EN" and e["type"][1] == "-")]
        tr = [t for t in tr if t]
        if tr:
            return tr
    return None


async def lookup(word: str, pos: str):
    """{lang: [варианты]} человеческих переводов слова (ru/ukr/pl/lt/ar + en-бонус),
    либо {} (нет в Lexin / сеть легла). ~1.5-3с: разведчик + 4 словаря параллельно."""
    key = (word or "").strip().lower()
    if not key:
        return {}
    try:
        ru_data = await asyncio.to_thread(_fetch, key, DICTS[0][0])
    except Exception as e:
        logger.warning(f"lexin '{key}': разведчик упал ({e}) — фолбэк на LLM")
        return {}
    out = {}
    ru = _parse(ru_data, key, pos, "Ru")
    if ru:
        out["ru"] = norm_translations(ru)
    en = _parse(ru_data, key, pos, "B")          # английский — бонус из ru-ответа
    if en:
        out["en"] = norm_translations(en)
    if not out:
        return {}                                 # леммы нет в общей базе — остальные не спрашиваем

    async def one(slug, prefix, lang):
        try:
            data = await asyncio.to_thread(_fetch, key, slug)
            tr = _parse(data, key, pos, prefix)
            return lang, (norm_translations(tr) if tr else None)
        except Exception:
            return lang, None

    results = await asyncio.gather(*(one(s, p, l) for s, p, l in DICTS[1:]))
    for lang, tr in results:
        if tr:
            out[lang] = tr
    logger.info(f"lexin '{key}'/{pos}: человеческие переводы для {sorted(out)}")
    return out
