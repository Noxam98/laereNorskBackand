"""Канонические части речи (англ. ключи POS_KEYS) + нормализация написаний.

История дубля: исходный корпус (≈3600 сущ. и пр., импорт 14–20 июня) размечен НОРВЕЖСКИМИ
ярлыками (substantiv/adjektiv/preposisjon…), а всё, что генерит/классифицирует LLM-пайплайн
(generate_words, pos_loop — схема POS_KEYS) и юзеры, использует АНГЛИЙСКУЮ схему (noun/adjective…).
Отсюда два написания. Канон — английский; normalize_pos сводит любой вариант к POS_KEYS, чтобы
движок обучения не зависел от написания в данных."""

# Канонические части речи (bokmål). Одно слово — одна часть речи (омографы → самое частотное).
POS_KEYS = ["noun", "verb", "adjective", "adverb", "preposition",
            "conjunction", "pronoun", "determiner", "numeral", "interjection", "phrase"]

# Написания (норвежские / сокращения / варианты) → канонический англ. ключ.
_POS_ALIASES = {
    "substantiv": "noun", "subst": "noun", "noun": "noun",
    "verb": "verb", "verbum": "verb",
    "adjektiv": "adjective", "adj": "adjective", "adjective": "adjective",
    "adverb": "adverb", "adverbium": "adverb", "adv": "adverb",
    "preposisjon": "preposition", "prep": "preposition", "preposition": "preposition",
    "konjunksjon": "conjunction", "subjunksjon": "conjunction",
    "konjunktion": "conjunction", "conjunction": "conjunction",
    "pronomen": "pronoun", "pronoun": "pronoun",
    "determinativ": "determiner", "determiner": "determiner", "artikkel": "determiner",
    "tallord": "numeral", "numeral": "numeral",
    "interjeksjon": "interjection", "interjection": "interjection",
    "phrase": "phrase", "uttrykk": "phrase", "frase": "phrase", "frase": "phrase",
}


def normalize_pos(pos):
    """Любое написание части речи → канонический англ. ключ (POS_KEYS). Неизвестное/пустое — как есть."""
    p = (pos or "").strip().lower()
    return _POS_ALIASES.get(p, p)


# Служебные части речи (для рампы): предлог/союз/местоимение/детерминатив. Функциональные наречия
# (ikke/opp/ut…) и частица «å» определяются ОТДЕЛЬНО по слову — см. db.learning.is_function_word.
FUNCTION_POS = {"preposition", "conjunction", "pronoun", "determiner"}
