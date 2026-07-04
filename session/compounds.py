"""Разблокировка составных слов (sammensetning) через изучение основ — ЧИСТАЯ логика подбора.

Модель (обсуждена 4.07): композит открывается, когда выучены ОБЕ его основы; учим рычажные
корни раньше собранных из них слов, а комплементы подтягиваем под сборку. Не жёсткий замок —
приоритеты в авто-доборе, всё саморазрешается. Вся математика тут, БЕЗ БД/asyncio: на вход
индекс частей + множества слов юзера + частоты, на выход — что открыто и как ранжировать новые.
IO (запросы, скрытый словарь) — в db/learning_suggest; индекс пула — воркер (autofill).

index — список записей пул-композитов: {"pool_id", "no", "forledd", "etterledd", "freq"}.
Множества (mastered/have/learnable) — norwegian-строки. Частоты — zipf (0..8).
"""

# Веса скоринга — частото-эквивалентные (в единицах zipf); тюнятся без релиза при нужде.
W_LEVERAGE = 0.3      # за каждый ЦЕННЫЙ композит, который основа открывает (охват корня); капим
W_COMPLETE = 1.0      # за каждый композит, который слово ДОСТРАИВАЕТ (другая часть уже известна)
W_DEFER = 3.0         # штраф композиту, чьи СТОЯЩИЕ основы ещё не выучены (корни раньше)
LEVERAGE_CAP = 8      # потолок вклада рычага (супер-корень не забивает всё)
VALUABLE_FREQ = 2.5   # композит «ценный», если freq ≥ этого (rare-junk не раздувает охват)
ROOT_MIN_LEVERAGE = 2  # основа «стоящая» для придержки — открывает ≥ N ценных композитов


def build_features(index):
    """Свёртка индекса → признаки на слово (для скоринга). Чистая.
    word -> {root_count: ценных композитов от этой основы, parts_of: [(комплемент, freq, no)],
             is_compound: bool, parts: (forledd, etterledd) | None}."""
    feat = {}

    def ensure(w):
        return feat.setdefault(w, {"root_count": 0, "parts_of": [], "is_compound": False, "parts": None})

    for c in index:
        no, fl, el, f = c["no"], c["forledd"], c["etterledd"], (c.get("freq") or 0.0)
        ce = ensure(no)
        ce["is_compound"] = True
        ce["parts"] = (fl, el)
        for part, other in ((fl, el), (el, fl)):
            pe = ensure(part)
            if f >= VALUABLE_FREQ:
                pe["root_count"] += 1
            pe["parts_of"].append((other, f, no))
    return feat


def leverage(word, feat):
    """Рычаг основы = число ЦЕННЫХ композитов, которые она открывает."""
    e = feat.get(word)
    return e["root_count"] if e else 0


def eligible_unlocks(index, mastered, have):
    """Композиты, ОТКРЫТЫЕ к вводу: обе части mastered, а самого слова у юзера ещё нет."""
    return [c for c in index
            if c["forledd"] in mastered and c["etterledd"] in mastered and c["no"] not in have]


def complete_count(word, feat, mastered):
    """Сколько ЦЕННЫХ композитов слово достроит: оно — часть, комплемент уже mastered."""
    e = feat.get(word)
    if not e:
        return 0
    return sum(1 for other, f, _no in e["parts_of"] if f >= VALUABLE_FREQ and other in mastered)


def is_deferred(word, feat, mastered, learnable):
    """Композит стоит придержать, если он составной И хотя бы одна его СТОЯЩАЯ основа
    (реальная лемма пула ∈ learnable + рычажная) ещё не выучена. Основы-неслова (bio-) и
    безрычажные (разовый редкий корень) НЕ придерживают — предохранитель от абсурда."""
    e = feat.get(word)
    if not e or not e["is_compound"] or not e["parts"]:
        return False
    for part in e["parts"]:
        if part in mastered:
            continue
        if part in learnable and leverage(part, feat) >= ROOT_MIN_LEVERAGE:
            return True
    return False


def score_new(word, freq, feat, mastered, learnable):
    """Приоритет нового слова = база(частота) + рычаг − придержка + буст комплемента."""
    base = freq or 0.0
    lev = min(leverage(word, feat), LEVERAGE_CAP)
    comp = complete_count(word, feat, mastered)
    deferred = is_deferred(word, feat, mastered, learnable)
    return base + W_LEVERAGE * lev + W_COMPLETE * comp - (W_DEFER if deferred else 0.0)


def rank_new(candidates, *, feat, mastered, learnable):
    """candidates: [(word, freq)] → слова по убыванию приоритета. Стабильно (равенство —
    по исходному порядку, т.е. по прежней частотной раскладке)."""
    scored = [(score_new(w, f, feat, mastered, learnable), i, w)
              for i, (w, f) in enumerate(candidates)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [w for _s, _i, w in scored]
