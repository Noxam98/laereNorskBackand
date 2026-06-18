"""LLM-слой проекта (API к Gemini). Разнесён по ответственностям:
  settings   — env: эндпоинты, пулы ключей, профили моделей с бюджетами
  quota      — ключи/квота: round-robin ключей, бюджеты, учёт, key-free хелперы
  client     — транспорт: клиенты, failover на 429, ask_json/ask_model/embed_text/embed_texts
  schemas    — темы/уровни/языки и JSON-схемы ответов
  embeddings — смысловой текст, хранение векторов, косинус/ANN, досчёт эмбеддингов
  words      — генерация/уточнение/сохранение слов

Ключи не покидают quota/client — остальной код зовёт key-free функции отсюда.
Реэкспорт ниже сохраняет привычные `from llm import X` / `llm.X`."""

from .settings import (
    LLM_MODEL, EMBED_MODEL, LLM_API_KEY, EMBED_API_KEY, LLM_API_KEYS, EMBED_API_KEYS,
)
from .quota import (
    text_enabled, embed_enabled, today, key_counts,
    text_available, embed_available, COOLDOWN_SEC,
)
from .client import (
    get_client, get_embed_client, ask_json, ask_model, embed_text, embed_texts, extract_json,
)
from .schemas import (
    TOPIC_TAGS, TOPIC_KEYS, CEFR_LEVELS, LANG_NAMES,
    WORDS_SCHEMA, DESC_SCHEMA, DIFF_SCHEMA, CLASSIFY_SCHEMA, REVIEW_SCHEMA,
    DESCRIBE_BATCH_SCHEMA, TRANSLATE_BATCH_SCHEMA, REFINE_SCHEMA,
    NOUN_FORMS_SCHEMA, VERB_FORMS_SCHEMA, ADJ_FORMS_SCHEMA,
    POS_REFINE_SCHEMA, POS_KEYS,
)
from .embeddings import (
    semantic_embed_text, encode_emb, decode_emb,
    rank_by_similarity, ranked_pool, ensure_embedding, ensure_embeddings,
)
from .words import (
    normalize_word_item, apply_item_meta, refine_translations, generate_words, persist_pool,
)
