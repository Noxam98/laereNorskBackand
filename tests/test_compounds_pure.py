"""ЧИСТАЯ логика разблокировки композитов (session/compounds.py) — без БД.

Замки на согласованную модель: рычаг корня, «корни раньше композитов» (soft-defer),
буст комплемента, eligibility, предохранитель на не-слова/безрычажные основы.
"""
from session import compounds as C

# Индекс: kjøleskap/kjøledisk/kjølevæske из корня kjøle; skap ещё в klesskap.
# freq — zipf; ценные (≥2.5): kjøleskap, kjøledisk, klesskap; kjølevæske — нет (rare).
INDEX = [
    {"pool_id": 1, "no": "kjøleskap", "forledd": "kjøle", "etterledd": "skap", "freq": 4.0},
    {"pool_id": 2, "no": "kjøledisk", "forledd": "kjøle", "etterledd": "disk", "freq": 3.0},
    {"pool_id": 3, "no": "kjølevæske", "forledd": "kjøle", "etterledd": "væske", "freq": 1.5},
    {"pool_id": 4, "no": "klesskap", "forledd": "klær", "etterledd": "skap", "freq": 3.5},
]
FEAT = C.build_features(INDEX)
# что реально существует как слово-лемма в пуле (для предохранителя)
LEARNABLE = {"kjøle", "skap", "disk", "væske", "klær", "kjøleskap", "kjøledisk", "klesskap"}


def test_leverage_counts_valuable_compounds():
    assert C.leverage("kjøle", FEAT) == 2      # kjøleskap+kjøledisk ценные, kjølevæske — нет
    assert C.leverage("skap", FEAT) == 2       # kjøleskap + klesskap
    assert C.leverage("væske", FEAT) == 0      # только в редком kjølevæske


def test_root_ranks_before_its_compound():
    """kjøle (корень) обгоняет kjøleskap, пока kjøle не выучен (soft-defer)."""
    cands = [("kjøleskap", 4.0), ("kjøle", 2.0)]   # у композита частота ВЫШЕ
    ranked = C.rank_new(cands, feat=FEAT, mastered=set(), learnable=LEARNABLE)
    assert ranked == ["kjøle", "kjøleskap"]        # рычаг+придержка перевесили частоту


def test_complement_boost_pulls_missing_part_forward():
    """Знаем kjøle → skap подтягивается вперёд (достроит kjøleskap+…), обгоняя равночастотное."""
    cands = [("annet", 3.0), ("skap", 3.0)]        # одинаковая база
    ranked = C.rank_new(cands, feat=FEAT, mastered={"kjøle"}, learnable=LEARNABLE)
    assert ranked[0] == "skap"


def test_eligible_unlock_needs_both_parts_mastered():
    assert [c["no"] for c in C.eligible_unlocks(INDEX, {"kjøle"}, set())] == []   # одна часть
    got = [c["no"] for c in C.eligible_unlocks(INDEX, {"kjøle", "skap"}, set())]
    assert got == ["kjøleskap"]                    # обе части → открыт
    # уже в словаре юзера → не предлагаем
    assert C.eligible_unlocks(INDEX, {"kjøle", "skap"}, {"kjøleskap"}) == []


def test_defer_guardrail_skips_nonword_and_low_leverage_roots():
    """Композит НЕ придерживаем за основой-не-словом или безрычажной."""
    idx = [{"pool_id": 9, "no": "biologi", "forledd": "bio", "etterledd": "logi", "freq": 4.0}]
    feat = C.build_features(idx)
    # bio/logi не в learnable (не слова) → biologi не придержан → идёт по своей частоте
    assert C.is_deferred("biologi", feat, set(), learnable={"biologi"}) is False
    # даже если бы были словами, рычаг < ROOT_MIN → не придерживаем
    assert C.leverage("bio", feat) < C.ROOT_MIN_LEVERAGE


def test_mastered_part_not_deferred():
    """Обе стоящие основы выучены → композит больше не придержан (готов всплыть)."""
    assert C.is_deferred("kjøleskap", FEAT, {"kjøle", "skap"}, LEARNABLE) is False
    assert C.is_deferred("kjøleskap", FEAT, {"kjøle"}, LEARNABLE) is True   # skap ещё нет
