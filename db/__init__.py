"""Слой данных, разнесённый по доменам. Реэкспорт — чтобы `from db import X`
работал как раньше.
  core         — соединение, схема (init_db), нормализация
  users        — пользователи
  pool         — общий пул слов + темы/уровни + поиск
  cache        — кэш запросов генерации + дневной учёт обращений
  dictionaries — словари пользователя и слова в них
"""
from .core import (
    DATABASE_URL, init_db, normalize_word, normalize_query,
    SQLITE_VEC_OK, vec_upsert, vec_delete, vec_nearest_rows,
)
from .users import (
    get_user, create_user, set_user_theme, set_user_game_prefs, set_user_current_dict,
    get_user_by_google_sub, create_google_user, set_user_google, clear_user_google,
    set_user_password, set_user_name, set_online_prefs, save_match,
)
from .pool import (
    get_or_create_pool, get_pool_id, get_pool_by_id,
    get_pool_tts, set_pool_tts, set_pool_embedding, get_pool_embeddings_raw,
    pool_missing_embedding, pool_missing_tts, tr_tts_pending, mark_tr_tts_done,
    translate_pending, mark_translate_done, update_pool_translate,
    sem_embed_pending, mark_sem_embed, get_pool_sample, get_pool_letter, get_pool_duel_words,
    set_pool_meta, pool_missing_meta, pool_missing_description, get_pool_topics_counts, get_pool_level_counts, get_pool_facets, get_pool_meta,
    get_pool_candidates, set_pool_description, delete_pool_word,
    get_pool_list, get_pool_ids, search_pool, get_pool_stats,
    set_pool_forms, pos_missing_forms, set_pool_pos, pos_uncategorized, clear_nonformable_forms,
)
from .cache import (
    get_cached_query, cache_query, set_cached_query, clear_query_cache, get_usage, get_usage_like, incr_usage,
)
from .dictionaries import (
    create_dictionary, delete_dictionary, add_word_to_dict, delete_dict_word,
    move_dict_word, set_word_override, record_result, get_dict_word, get_user_data,
    get_user_quiz_words,
)
