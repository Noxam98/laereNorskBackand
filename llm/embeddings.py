"""Эмбеддинг-домен: смысловой текст слова, бинарное хранение векторов (float16),
косинусный поиск и ANN, best-effort досчёт эмбеддингов. Запросы к провайдеру — через
client.embed_text/embed_texts (ключей здесь не видно)."""
import json
import asyncio
import numpy as np
from db import get_pool_by_id, set_pool_embedding, vec_nearest_rows, get_pool_candidates
from .settings import EMBED_API_KEYS, EMBED_API_KEY
from .client import embed_text, embed_texts
from langs import LANG_CODES

_EMB_LANGS = LANG_CODES   # derive из реестра — эмбеддинг учитывает все языки перевода (в т.ч. lv/ar)


def semantic_embed_text(data):
    """Текст для эмбеддинга по СМЫСЛУ: норвежское слово + все переводы.
    Так вектор отражает значение, а не написание (соседи — по смыслу)."""
    data = data or {}
    tr = data.get("translate", {}) or {}
    parts = [data.get("word") or (tr.get("no") or [""])[0]]
    for l in _EMB_LANGS:
        parts.extend(v for v in (tr.get(l) or []) if v)
    return ", ".join(p for p in parts if p).strip()


# --- Эмбеддинги: бинарное хранение (float16) + матричный косинус (мало RAM/CPU) ---
def encode_emb(vec):
    return np.asarray(vec, dtype=np.float16).tobytes()


def decode_emb(v):
    if not v:
        return None
    if isinstance(v, (bytes, bytearray)) and not v[:1] == b"[":
        return np.frombuffer(v, dtype=np.float16).astype(np.float32)
    try:
        return np.asarray(json.loads(v), dtype=np.float32)  # legacy JSON
    except Exception:
        return None


def rank_by_similarity(target_raw, cands):
    """Вернуть cands (с эмбеддингом) по убыванию близости к target. None — у target нет вектора."""
    tv = decode_emb(target_raw)
    if tv is None:
        return None
    rows, vecs = [], []
    for c in cands:
        ev = decode_emb(c.get("embedding"))
        if ev is not None and ev.shape == tv.shape:
            rows.append(c); vecs.append(ev)
    if not rows:
        return []
    M = np.vstack(vecs)
    sims = (M @ tv) / (np.linalg.norm(M, axis=1) * (np.linalg.norm(tv) + 1e-9) + 1e-9)
    return [rows[i] for i in np.argsort(-sims)]


async def ranked_pool(target_raw, exclude_norwegian, n):
    """Ближайшие по смыслу слова пула [{norwegian, data(dict)}], исключая exclude.
    Использует ANN-индекс (sqlite-vec) если доступен, иначе brute-force в отдельном потоке."""
    if not target_raw:
        return []
    rows = await vec_nearest_rows(target_raw, n + 5)  # None — индекс недоступен
    if rows is not None:
        out = []
        for r in rows:
            if r["norwegian"] == exclude_norwegian:
                continue
            out.append({"norwegian": r["norwegian"], "data": json.loads(r["data"]) if r["data"] else {}})
            if len(out) >= n:
                break
        return out
    # фолбэк: перебор всех кандидатов (CPU — в треде, чтобы не блокировать event loop)
    cands = [c for c in await get_pool_candidates() if c["norwegian"] != exclude_norwegian and c.get("embedding")]
    ranked = await asyncio.to_thread(rank_by_similarity, target_raw, cands)
    if not ranked:
        return []
    return [{"norwegian": c["norwegian"], "data": c["data"]} for c in ranked[:n]]


async def ensure_embedding(pool_id, norwegian):
    """Best-effort: посчитать и сохранить эмбеддинг слова по смыслу, если его ещё нет."""
    if not EMBED_API_KEY:
        return
    p = await get_pool_by_id(pool_id)
    if not p or p.get("embedding"):
        return
    vec = await embed_text(semantic_embed_text(p["data"]) or norwegian)
    if vec:
        await set_pool_embedding(pool_id, encode_emb(vec))


async def ensure_embeddings(items):
    """Best-effort эмбеддинг ПАЧКОЙ (один запрос) для слов без вектора.
    items: [(pool_id, data_dict)]. Тише в ленте и экономит квоту против поштучного."""
    if not EMBED_API_KEYS or not items:
        return
    pend = []  # (pid, текст)
    for pid, data in items:
        if not pid:
            continue
        p = await get_pool_by_id(pid)
        if p and not p.get("embedding"):
            text = semantic_embed_text(data if isinstance(data, dict) else {}) or (data if isinstance(data, str) else "")
            if text:
                pend.append((pid, text))
    if not pend:
        return
    vecs = await embed_texts([t for _, t in pend])
    if vecs and len(vecs) == len(pend):
        for (pid, _), vec in zip(pend, vecs):
            await set_pool_embedding(pid, encode_emb(vec))
