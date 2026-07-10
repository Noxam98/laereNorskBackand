"""Клетки рампы и вид рампы слова (RampKind). Чистый модуль без зависимостей.

Раньше «вид рампы» определялся ССЫЛОЧНЫМ сравнением списков (`cells is REQUIRED_CELLS`)
— рефактор required_cells в копию списка молча ломал классификацию контентных слов
(включая откладывание аудио в слуховую партию). Теперь вид — явное значение RampKind,
а наборы клеток — неизменяемые кортежи-константы.
"""

# Виды рампы
CONTENT = "content"          # обычные слова: choice×2 → build → input
FUNC_CLOZE = "func_cloze"    # служебные при включённом cloze: 3 cloze-предложения
FUNC_CHOICE = "func_choice"  # служебные при выключенном cloze: только выбор
PHRASE = "phrase"            # устойчивые выражения: выбор → порядок слов → сборка из букв → ввод

AUDIO_CELL = "choice_no2int"   # аудио-подтверждение (при audio ВКЛ — в слуховую партию)

REQUIRED_CELLS = ("choice_int2no", "choice_no2int", "build_int2no", "input_int2no")
FUNC_CELLS = ("cloze_1", "cloze_2", "cloze_3")
FUNC_CELLS_CHOICE = ("choice_int2no", "choice_no2int")
# Рампа фраз: выбор перевода → порядок слов → СБОРКА ИЗ БУКВ (тот же BuildGame, что у слов;
# клавиатура сама даёт пробел ровно столько раз, сколько его во фразе) → свободный ввод.
PHRASE_CELLS = ("choice_no2int", "order_int2no", "build_int2no", "input_int2no")
# Прежняя рампа фраз (без сборки). Grandfathering: кто прошёл ЕЁ целиком — остаётся выученным,
# новая ступень его не «разучивает» (ввод по памяти тяжелее сборки из букв). См. srs/status.is_mastered.
PHRASE_CELLS_LEGACY = ("choice_no2int", "order_int2no", "input_int2no")
ALL_CELLS = REQUIRED_CELLS + FUNC_CELLS + ("order_int2no",)   # order — клетка фраз (демоут её тоже чистит)

_CELLS_OF = {
    CONTENT: REQUIRED_CELLS,
    FUNC_CLOZE: FUNC_CELLS,
    FUNC_CHOICE: FUNC_CELLS_CHOICE,
    PHRASE: PHRASE_CELLS,
}


def ramp_kind(*, phrase_playable: bool, function_word: bool, cloze_enabled: bool) -> str:
    """Вид рампы слова. Вся «грязная» классификация (парсинг data, словарь служебных)
    остаётся в фасаде db/learning.required_cells — сюда приходят готовые булевы."""
    if phrase_playable:
        return PHRASE
    if function_word:
        return FUNC_CLOZE if cloze_enabled else FUNC_CHOICE
    return CONTENT


def cells_of(kind: str) -> tuple:
    """Клетки рампы данного вида (неизменяемый кортеж)."""
    return _CELLS_OF[kind]
