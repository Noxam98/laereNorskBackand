"""Единый реестр языков перевода/интерфейса (бэкенд).

ДОБАВИТЬ ЯЗЫК = ОДНА запись в LANGUAGES здесь. Все списки/множества/циклы/валидация
и языковые куски промптов derive отсюда — больше ничего по коду править не нужно
(парный фронт-реестр: src/interface/languages.js).

Поля записи:
  code     — ключ в word_pool.translate и в gamePrefs.lang (фронт-код; украинский = "ukr")
  name     — название языка по-русски (для промптов «переведи на …»)
  name_in  — форма «на …ском» / эндоним (для diff-промпта «объясни разницу на …»)
  tts      — код голоса озвучки перевода (украинский = "uk")
"""

LANGUAGES = [
    {"code": "ru",  "name": "русский",    "name_in": "русском",    "tts": "ru"},
    {"code": "ukr", "name": "украинский", "name_in": "украинском", "tts": "uk"},
    {"code": "en",  "name": "английский", "name_in": "English",    "tts": "en"},
    {"code": "pl",  "name": "польский",   "name_in": "polskim",    "tts": "pl"},
    {"code": "lt",  "name": "литовский",  "name_in": "lietuvių",   "tts": "lt"},
    {"code": "lv",  "name": "латышский",  "name_in": "latviešu",   "tts": "lv"},
]

# --- производные (НЕ редактировать вручную) ---
LANG_CODES = [l["code"] for l in LANGUAGES]                       # ["ru","ukr","en","pl","lt","lv"]
LANG_SET = set(LANG_CODES)
LANG_NAMES = {l["code"]: l["name"] for l in LANGUAGES}            # для промптов (рус. названия)
DIFF_LANG_NAMES = {l["code"]: l["name_in"] for l in LANGUAGES}   # форма «на …ском»
TTS_LANGS = {l["tts"] for l in LANGUAGES}                         # коды голосов озвучки ({"uk",…})

# Готовые строки для промптов — чтобы добавление языка не требовало правки текстов промптов.
LANGS_CSV = ", ".join(LANG_CODES)                                # "ru, ukr, en, pl, lt, lv"
LANGS_SLASH = "/".join(LANG_CODES)                               # "ru/ukr/en/pl/lt/lv"
LANG_NAMES_CSV = ", ".join(LANG_NAMES[c] for c in LANG_CODES)    # "русский, украинский, …"
# Пример блока translate для промпта генерации слов (ключи — строго по реестру).
TRANSLATE_EXAMPLE = ",\n".join(f'            "{c}": ["вариант 1", "вариант 2"]' for c in LANG_CODES)
# Пример блока описаний для промпта description_task (ключи — по реестру).
DESC_EXAMPLE = ",\n".join(f'    "{c}": "описание на этом языке"' for c in LANG_CODES)
