"""Юнит-тесты правок морфологии/fuzzy (см. задачу «Чинишь правила словоизменения»).

Правила — источник ДЕТЕКТА нерегулярности и генерации/фильтра дистракторов
(истина форм = БД/ordbank). Тесты — чистые проверки значений.
"""
from morphology import noun, adjective, verb
from morphology import (regular_indef_pl, regular_def_pl, is_irregular_noun,
                        regular_neuter, is_irregular_adj,
                        predict_present, predict_weak_class, regular_past, regular_perfect,
                        all_weak_pasts, all_weak_perfects, is_irregular_verb)
from fuzzy import word_forms, fuzzy_match, normalize


# ── #1: utlending — это -ing, а не агентив на -er ────────────────────────────

def test_utlending_regular_plural():
    # дефолт +er верен: utlending→utlendinger (а не *utlendinge)
    assert regular_indef_pl("utlending", "en") == "utlendinger"
    assert regular_def_pl("utlending", "en") == "utlendingene"


def test_utlending_not_in_agentive_list():
    assert "utlending" not in noun.AGENTIVE_ER
    # настоящие агентивы на -er остались
    assert "lærer" in noun.AGENTIVE_ER
    assert "innbygger" in noun.AGENTIVE_ER


def test_utlending_not_flagged_irregular():
    ok = is_irregular_noun("utlending", "en",
                           {"indef_pl": "utlendinger", "def_sg": "utlendingen",
                            "def_pl": "utlendingene"})[0]
    assert ok is False


# ── #2: дегеминация удвоенного согласного перед нейтральным -t ────────────────

def test_neuter_degeminates_double_consonant():
    assert regular_neuter("stygg") == "stygt"
    assert regular_neuter("trygg") == "trygt"
    assert regular_neuter("snill") == "snilt"
    assert regular_neuter("tøff") == "tøft"
    assert regular_neuter("vill") == "vilt"
    assert regular_neuter("frekk") == "frekt"
    # -nn регрессия (частный случай общего правила)
    assert regular_neuter("grønn") == "grønt"
    assert regular_neuter("tynn") == "tynt"


def test_neuter_keep_double_exceptions():
    # full сохраняет удвоение: fullt (не *fult)
    assert regular_neuter("full") == "fullt"


def test_neuter_dd_stem_unchanged():
    # основа на -dd не добавляет -t: redd→redd (не *redt/*reddt)
    assert regular_neuter("redd") == "redd"


def test_stygg_not_flagged_irregular_by_neuter():
    ok = is_irregular_adj("stygg", {"neuter": "stygt", "plural": "stygge",
                                    "comparative": "styggere", "superlative": "styggest"})[0]
    assert ok is False


# ── #3: junk-маркеры LLM не попадают в принимаемые формы ──────────────────────

def test_word_forms_excludes_junk():
    forms = {"present": "tier", "past": "n/a", "perfect": "ingen",
             "extra": "-", "pos": "verb"}
    got = word_forms("tie", forms)
    assert got == ["tie", "tier"]
    lowered = {w.lower() for w in got}
    for junk in ("n/a", "na", "ingen", "-", "none", "null"):
        assert junk not in lowered


def test_junk_not_accepted_on_exam():
    # раньше «na» проходило как опечатка «n/a» из forms → теперь junk отфильтрован
    forms = word_forms("bra", {"comparative": "n/a", "superlative": "-", "pos": "adj"})
    assert fuzzy_match("na", forms) is False
    assert fuzzy_match("ingen", forms) is False


# ── #4: односложные глаголы на -e с гласной основой (tie/skje) ────────────────

def test_vowel_stem_e_forms():
    assert predict_weak_class("tie") == (4, "tidde", "har tidd")
    assert predict_weak_class("skje") == (4, "skjedde", "har skjedd")
    assert regular_past("tie") == "tidde"
    assert regular_perfect("tie") == "har tidd"
    assert regular_past("skje") == "skjedde"
    assert regular_perfect("skje") == "har skjedd"


def test_vowel_stem_e_present():
    assert predict_present("tie") == "tier"
    assert predict_present("skje") == "skjer"


def test_vowel_stem_e_allowed_sets():
    assert all_weak_pasts("tie") == {"tidde"}
    assert all_weak_perfects("tie") == {"tidd"}


def test_vowel_stem_e_not_flagged_irregular():
    ok = is_irregular_verb("tie", {"present": "tier", "past": "tidde",
                                   "perfect": "har tidd"})[0]
    assert ok is False


def test_regular_weak_verbs_still_work():
    # не сломали регулярные слабые
    assert regular_past("spise") == "spiste"
    assert regular_past("kaste") == "kastet"
    assert regular_past("leve") == "levde"
    assert regular_past("bo") == "bodde"


# ── #5: гигиена списков ──────────────────────────────────────────────────────

def test_list_hygiene():
    assert "vær" not in noun.UNCOUNTABLE_NOUNS
    assert "vær" not in noun.SYNCOPE_NOUNS
    assert "tabell" not in noun.SYNCOPE_NOUNS
    assert "fanger" not in noun.SYNCOPE_NOUNS
    assert "barnslig" not in adjective.ADJ_SK_NO_T
    # barnslig всё равно без -t (обрабатывается веткой -ig)
    assert regular_neuter("barnslig") == "barnslig"
