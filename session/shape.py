"""Единая форма элемента сессии (Этап 6): один конструктор на все билдеры.

До этапа элемент строился ЧЕТЫРЬМЯ независимыми dict-литералами (дневная сессия,
слуховая партия, живой добор next-cards, грамм-тир form/overlay) — поля тихо
дрейфовали между /learning/session и /learning/next-cards. Теперь базовые ключи
присутствуют ВСЕГДА (grammar/own_options — явные False у контентных элементов),
а дополнительные — только из замороженного whitelist: новый ключ = осознанная
правка EXTRA_KEYS и теста, а не молчаливый дрейф формы.

own_options=True — элемент несёт СВОИ варианты (морфологические подмены
грамм-тира): _attach_choice_options такие не трогает (Этап 7 переводит его
фильтр с тега grammar на own_options). Конструктор страхует контракт:
choice-элемент с own_options обязан принести непустые options.
"""

# Базовая форма: есть у КАЖДОГО элемента любой сессии/добора.
BASE_KEYS = ("pool_id", "no", "translate", "part_of_speech", "forms",
             "mode", "direction", "step", "repeat", "grammar", "own_options")

# Замороженный union дополнительных ключей всех билдеров (field-diff Этапа 6).
EXTRA_KEYS = frozenset((
    "gloss", "example",              # карточка слова (дневная сессия / next-cards)
    "listen",                        # слуховая партия: проигрывать аудио, текст скрыт
    "cloze",                         # cloze-предложение служебного слова
    "options", "distractors",        # варианты выбора (свои у грамм-тира, патч у контентных)
    "form_track", "stage",           # трек форм: маршрутизация ответа + ступень рампы клетки
    "prompt", "target", "reveal",    # параметризованный контракт грамм-элемента
    "scoring",                       # правила зачёта ввода (typoForgive)
    "compound",                      # разбор составного слова (для «ага» на карточке-знакомстве)
))


def make_element(*, pool_id, no, translate, part_of_speech, forms, mode,
                 direction, step, repeat, grammar=False, own_options=False, **extra):
    """Элемент сессии. Обязательные поля — именованными аргументами (забыть нельзя),
    дополнительные — из EXTRA_KEYS (иначе ValueError: дрейф формы ловится в тестах
    билдера, а не багрепортом с фронта)."""
    unknown = set(extra) - EXTRA_KEYS
    if unknown:
        raise ValueError(f"неизвестные ключи элемента сессии: {sorted(unknown)}")
    if own_options and mode == "choice" and not extra.get("options"):
        raise ValueError("own_options=True у choice-элемента требует непустых options")
    el = {"pool_id": pool_id, "no": no, "translate": translate,
          "part_of_speech": part_of_speech, "forms": forms, "mode": mode,
          "direction": direction, "step": step, "repeat": repeat,
          "grammar": grammar, "own_options": own_options}
    el.update(extra)
    return el
