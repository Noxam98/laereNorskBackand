"""embcache: резидентный кеш эмбеддингов в RAM → дистракторы живым matvec, без sqlite-KNN.
Проверяем: ближайший по вектору первым; новое слово (update_vec) сразу попадает в кандидаты;
пере-эмбеддинг меняет соседство. Источник правды — БД; ensure_loaded строит матрицу из неё."""
import numpy as np
import importlib

import embcache
from db.core import EMBED_DIM
from db.pool import get_or_create_pool
from db.pool_queues import set_pool_embedding
from llm.embeddings import encode_emb


def _vec(*nonzero):
    v = np.zeros(EMBED_DIM, dtype=np.float32)
    for i, x in nonzero:
        v[i] = x
    return encode_emb(v)


async def _seed(no, emb):
    pid = await get_or_create_pool(no, {"word": no, "translate": {"ru": [no]}, "part_of_speech": "noun"})
    await set_pool_embedding(pid, emb)   # синкает кеш, если он уже загружен
    return pid


def _reset_cache():
    importlib.reload(embcache)  # сбросить модульное состояние кеша между тестами


async def test_nearest_from_cache(fresh_db):
    _reset_cache()
    a = await _seed("a", _vec((0, 1.0)))
    b = await _seed("b", _vec((0, 0.99), (1, 0.01)))   # почти коллинеарен a
    await _seed("c", _vec((2, 1.0)))                    # ортогонален a/b
    await embcache.ensure_loaded(force=True)            # построить матрицу из БД
    nbr = await embcache.candidates_for([a, b], 45)
    assert nbr[a][0] == b                               # ближайший к a — b
    assert nbr[b][0] == a
    st = embcache.cache_stats()
    assert st["ready"] and st["loaded"] == 3 and st["dim"] == EMBED_DIM


async def test_new_word_visible_without_reload(fresh_db):
    _reset_cache()
    a = await _seed("a", _vec((0, 1.0)))
    await _seed("z", _vec((5, 1.0)))
    await embcache.ensure_loaded(force=True)
    # новое слово, очень близкое к a — set_pool_embedding должен синкнуть его в кеш (update_vec)
    d = await _seed("a2", _vec((0, 0.98), (1, 0.02)))
    nbr = await embcache.candidates_for([a], 45)
    assert d in nbr[a]                                  # старое слово «узнало» о новом без рестарта
    assert nbr[a][0] == d                               # и оно — ближайшее


async def test_reembed_changes_neighbors(fresh_db):
    _reset_cache()
    a = await _seed("a", _vec((0, 1.0)))
    b = await _seed("b", _vec((1, 1.0)))
    c = await _seed("c", _vec((0, 0.9), (1, 0.1)))
    await embcache.ensure_loaded(force=True)
    assert (await embcache.candidates_for([a], 45))[a][0] == c   # сперва ближе c
    await set_pool_embedding(b, _vec((0, 0.99), (1, 0.01)))      # b переехал ближе к a
    assert (await embcache.candidates_for([a], 45))[a][0] == b   # теперь ближе b
