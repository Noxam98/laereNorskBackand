"""Резидентный кеш эмбеддингов пула в ОЗУ — для подбора дистракторов БЕЗ живого sqlite-KNN.

Корень тормозов был: vec0/KNN по 3072-мерным векторам читал ~70МБ индекса с диска НА КАЖДЫЙ запрос,
а сборка сессии звала KNN по разу на каждый choice-элемент → 30-45с. Решение: держим всю матрицу
эмбеддингов (нормированную, float32 ≈72МБ — машина 512МБ, влезает) постоянно в RAM и считаем
ближайших одним numpy-matmul прямо в момент сборки (Q@M.T, миллисекунды, читает RAM а не диск).

Плюсы против предрасчёта: всегда свежо (матрица — текущая), новое слово сразу кандидат для ВСЕХ
(матвек идёт против полной матрицы), не нужен ни воркер, ни колонка, ни симметричные вставки.
Обновление инкрементально: новый вектор — append (копия ~7мс), пере-эмбеддинг — замена строки."""
import asyncio
import threading

import numpy as np

from db import load_pool_embeddings, get_pool_by_id
from db.core import _f16_to_f32_bytes

_lock = threading.Lock()
_M = None         # float32 N×D, строки нормированы (косинус = скалярное произведение)
_ids = None       # np.int64 N — id пула по строкам
_rowof = {}       # {pool_id: row}
_ready = False


def _norm_row(raw):
    """Сырой эмбеддинг (f16 bytes) → нормированный float32 вектор, или None у битых."""
    b = _f16_to_f32_bytes(raw, np)
    if b is None:
        return None
    v = np.frombuffer(b, dtype=np.float32).copy()
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _build(ids, embs):
    rows, mats = [], []
    for pid, raw in zip(ids, embs):
        v = _norm_row(raw)
        if v is not None:
            rows.append(int(pid)); mats.append(v)
    if not rows:
        return None, np.empty(0, dtype=np.int64), {}
    M = np.vstack(mats)
    idarr = np.asarray(rows, dtype=np.int64)
    return M, idarr, {pid: i for i, pid in enumerate(rows)}


async def ensure_loaded(force=False):
    """Загрузить матрицу в RAM (один раз на старте). Идемпотентно."""
    global _M, _ids, _rowof, _ready
    if _ready and not force:
        return
    ids, embs = await load_pool_embeddings()
    M, idarr, rowof = await asyncio.to_thread(_build, ids, embs)
    with _lock:
        _M, _ids, _rowof, _ready = M, idarr, rowof, True


def update_vec(pool_id, raw):
    """Поддержать кеш свежим при изменении эмбеддинга: замена строки (пере-эмбеддинг) или append
    (новое слово). Дёшево (in-place / одна копия). No-op, если кеш ещё не загружен — подхватится при
    первой полной загрузке. Вызывается из set_pool_embedding (поздним импортом, без цикла)."""
    global _M, _ids
    if not _ready:
        return
    v = _norm_row(raw)
    if v is None:
        return
    with _lock:
        row = _rowof.get(int(pool_id))
        if row is not None:
            _M[row] = v                                   # пере-эмбеддинг — замена на месте
        else:
            _M = np.vstack([_M, v])                        # новое слово — дописать строку
            _ids = np.append(_ids, np.int64(pool_id))
            _rowof[int(pool_id)] = _M.shape[0] - 1


def _nearest(M, ids, rowof, pool_ids, k):
    """CPU-часть (в потоке): {pool_id: [neighbor_id,...]} по убыванию близости, исключая себя."""
    present = [(p, rowof[p]) for p in pool_ids if p in rowof]
    if not present or M is None:
        return {}
    q_rows = np.asarray([r for _, r in present])
    sims = M[q_rows] @ M.T                                 # (len×N)
    kk = min(k + 1, M.shape[0])
    part = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    out = {}
    for i, (pid, _) in enumerate(present):
        cand = part[i]
        order = cand[np.argsort(-sims[i, cand])]
        out[pid] = [int(ids[j]) for j in order if int(ids[j]) != pid][:k]
    return out


async def candidates_for(pool_ids, k=45):
    """{pool_id: [neighbor_id,...]} ближайших по смыслу — для дистракторов. Живой matvec в RAM."""
    await ensure_loaded()
    with _lock:
        M, ids, rowof = _M, _ids, _rowof
    if M is None or not pool_ids:
        return {}
    return await asyncio.to_thread(_nearest, M, ids, rowof, list(pool_ids), k)


def cache_stats():
    """Для админки: сколько векторов в RAM, размерность, объём."""
    with _lock:
        if not _ready or _M is None:
            return {"loaded": 0, "dim": 0, "mb": 0, "ready": _ready}
        return {"loaded": int(_M.shape[0]), "dim": int(_M.shape[1]),
                "mb": round(_M.nbytes / 1e6, 1), "ready": True}
