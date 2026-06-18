"""Канонические темы/уровни, языки и JSON-схемы ответов (structured output).
Чистые данные — без логики и запросов."""

# Канонические темы-теги пула (стабильные ключи; UI-подписи — во фронтовом i18n).
# Значение — рус. подсказка для LLM-классификатора и генерации.
TOPIC_TAGS = {
    "family": "семья и родственники",
    "food": "еда, напитки, продукты",
    "home": "дом, мебель, быт, посуда",
    "work": "работа, профессии, офис, бизнес",
    "school": "школа, учёба, образование, наука",
    "travel": "путешествия, туризм, отдых, гостиница",
    "health": "здоровье, болезни, медицина, аптека",
    "body": "тело человека, органы чувств",
    "clothing": "одежда, обувь, аксессуары",
    "nature": "природа, ландшафт, растения, деревья",
    "animals": "животные, птицы, рыбы, насекомые",
    "weather": "погода, климат, времена года",
    "city": "город, здания, места, улицы",
    "transport": "транспорт, машины, дорога, движение",
    "shopping": "деньги, покупки, магазин, финансы",
    "time": "время, даты, дни, месяцы",
    "sport": "спорт, фитнес, игры",
    "hobby": "хобби, досуг, культура, искусство, музыка, кино",
    "technology": "технологии, компьютеры, интернет, гаджеты",
    "communication": "общение, речь, связь, приветствия",
    "emotions": "эмоции, чувства, черты характера",
    "holidays": "праздники, традиции, события",
    "society": "общество, политика, право, экономика",
    "other": "прочее, абстрактные понятия, качества, количества",
}
TOPIC_KEYS = list(TOPIC_TAGS.keys())
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]

# Названия языков перевода (ключ интерфейса → язык на русском для промпта).
LANG_NAMES = {"ru": "русский", "ukr": "украинский", "en": "английский", "pl": "польский", "lt": "литовский"}

# Схемы для гарантированного формата ответа (structured output / JSON-schema).
_STR_ARR = {"type": "array", "items": {"type": "string"}}

WORDS_SCHEMA = {
    "name": "words_response",
    "schema": {
        "type": "object",
        "properties": {
            "words": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string", "description": "норвежское слово, без артикля, нейтральная форма"},
                        "translate": {
                            "type": "object",
                            "properties": {"ru": _STR_ARR, "ukr": _STR_ARR, "en": _STR_ARR, "pl": _STR_ARR, "lt": _STR_ARR},
                        },
                        "part_of_speech": {"type": "string"},
                        "level": {"type": "string", "enum": CEFR_LEVELS},
                        "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_KEYS}},
                    },
                    "required": ["word", "translate", "part_of_speech"],
                },
            }
        },
        "required": ["words"],
    },
}

DESC_SCHEMA = {
    "name": "description_response",
    "schema": {
        "type": "object",
        "properties": {"ru": {"type": "string"}, "ukr": {"type": "string"}, "en": {"type": "string"}, "pl": {"type": "string"}, "lt": {"type": "string"}},
        "required": ["ru", "ukr", "en", "pl", "lt"],
    },
}

# Разница между двумя норвежскими словами (на языке пользователя).
DIFF_SCHEMA = {
    "name": "word_diff_response",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},   # суть различия одной фразой
            "when_a": {"type": "string"},     # когда употреблять первое слово
            "when_b": {"type": "string"},     # когда употреблять второе слово
            "example": {"type": "string"},    # короткий пример (норвежский + перевод)
        },
        "required": ["summary", "when_a", "when_b", "example"],
    },
}

# Ревью правки слова: одобрить/отклонить + причина + СТАНДАРТИЗОВАННОЕ слово (при апруве):
# исправленная орфография норвежского, часть речи, переводы на 5 языков.
REVIEW_SCHEMA = {
    "name": "edit_review_response",
    "schema": {
        "type": "object",
        "properties": {
            "approved": {"type": "boolean"},  # одобрить правку
            "reason": {"type": "string"},     # почему одобрено / что не так (на языке пользователя)
            "word": {                          # заполняется при approved=true (исправленный канон)
                "type": "object",
                "properties": {
                    "word": {"type": "string"},            # норвежское слово (bokmål, с исправленной опечаткой)
                    "part_of_speech": {"type": "string"},  # noun | verb | adjective | ...
                    "translate": {
                        "type": "object",
                        "properties": {
                            "ru": {"type": "array", "items": {"type": "string"}},
                            "ukr": {"type": "array", "items": {"type": "string"}},
                            "en": {"type": "array", "items": {"type": "string"}},
                            "pl": {"type": "array", "items": {"type": "string"}},
                            "lt": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
            },
        },
        "required": ["approved", "reason"],
    },
}

# Пакетная классификация слов: уровень CEFR + 1-3 темы из канонического списка.
CLASSIFY_SCHEMA = {
    "name": "classify_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "level": {"type": "string", "enum": CEFR_LEVELS},
                        "topics": {"type": "array", "items": {"type": "string", "enum": TOPIC_KEYS}, "minItems": 1, "maxItems": 3},
                    },
                    "required": ["word", "level", "topics"],
                },
            }
        },
        "required": ["results"],
    },
}

DESCRIBE_BATCH_SCHEMA = {
    "name": "describe_batch_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "ru": {"type": "string"}, "ukr": {"type": "string"}, "en": {"type": "string"},
                        "pl": {"type": "string"}, "lt": {"type": "string"},
                    },
                    "required": ["word", "ru", "ukr", "en", "pl", "lt"],
                },
            }
        },
        "required": ["results"],
    },
}

# Пакетный перевод: для каждого норвежского слова — варианты перевода на 5 языков.
TRANSLATE_BATCH_SCHEMA = {
    "name": "translate_batch_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "word": {"type": "string"},
                        "ru": _STR_ARR, "ukr": _STR_ARR, "en": _STR_ARR, "pl": _STR_ARR, "lt": _STR_ARR,
                    },
                    "required": ["word", "ru", "ukr", "en", "pl", "lt"],
                },
            }
        },
        "required": ["results"],
    },
}

# Канонические части речи (bokmål). Ключи — стабильные, англ.; UI-подписи во фронте.
POS_KEYS = ["noun", "verb", "adjective", "adverb", "preposition",
            "conjunction", "pronoun", "determiner", "numeral", "interjection", "phrase"]

# Переразметка части речи: для слов с пустой/нераспознанной part_of_speech.
POS_REFINE_SCHEMA = {
    "name": "pos_refine",
    "schema": {"type": "object", "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"word": {"type": "string"}, "part_of_speech": {"type": "string", "enum": POS_KEYS}},
        "required": ["word", "part_of_speech"],
    }}}, "required": ["results"]},
}


# --- Грамматические формы по части речи (structured output, батч) ---
def _forms_schema(name, fields):
    return {"name": name, "schema": {"type": "object", "properties": {"results": {"type": "array", "items": {
        "type": "object",
        "properties": {"word": {"type": "string"}, **{f: {"type": "string"} for f in fields}},
        "required": ["word"],
    }}}, "required": ["results"]}}


# Существительное: род + склонение.
NOUN_FORMS_SCHEMA = _forms_schema("noun_forms", ["gender", "def_sg", "indef_pl", "def_pl"])
# Глагол: спряжение (инфинитив = само слово).
VERB_FORMS_SCHEMA = _forms_schema("verb_forms", ["present", "past", "perfect"])
# Прилагательное: степени + согласование.
ADJ_FORMS_SCHEMA = _forms_schema("adj_forms", ["comparative", "superlative", "neuter", "plural"])


# Уточнение перевода группы слов (одинаковые/неточные переводы) на один язык.
REFINE_SCHEMA = {
    "name": "refine_translate_response",
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"word": {"type": "string"}, "translate": _STR_ARR},
                    "required": ["word", "translate"],
                },
            }
        },
        "required": ["results"],
    },
}
