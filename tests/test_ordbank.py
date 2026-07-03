"""Ordbank-слой форм: локальный дамп (lookup) и конвертер живого ordbøkene (_parse_uib).
Сеть не трогаем — API-ответы зафиксированы фикстурами."""
import json
import sqlite3

import pytest

from db import ordbank


@pytest.fixture
def bank(tmp_path, monkeypatch):
    p = tmp_path / "ordbank.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE forms (norwegian TEXT, pos TEXT, forms TEXT, PRIMARY KEY (norwegian, pos))")
    c.execute("INSERT INTO forms VALUES ('hus','noun', ?)",
              (json.dumps({"pos": "noun", "gender": "et", "def_sg": "huset"}),))
    c.commit(); c.close()
    monkeypatch.setattr(ordbank, "PATH", str(p))
    monkeypatch.setattr(ordbank, "_conn", None)
    yield
    monkeypatch.setattr(ordbank, "_conn", None)


def test_lookup_hit_and_miss(bank):
    f = ordbank.lookup("Hus", "noun")           # регистронезависимо
    assert f and f["gender"] == "et" and f["def_sg"] == "huset"
    assert ordbank.lookup("finnesikke", "noun") is None
    assert ordbank.lookup("hus", "verb") is None


def test_lookup_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ordbank, "PATH", str(tmp_path / "нет.db"))
    monkeypatch.setattr(ordbank, "_conn", None)
    assert ordbank.lookup("hus", "noun") is None   # нет файла — тихий None (LLM-фолбэк)


def _noun_article():
    """Скелет реального ответа ord.uib.no для skjermtid (дублет Masc/Fem)."""
    return {"lemmas": [{"lemma": "skjermtid", "paradigm_info": [
        {"tags": ["NOUN", "Masc"], "inflection": [
            {"tags": ["Sing", "Ind"], "word_form": "skjermtid"},
            {"tags": ["Sing", "Def"], "word_form": "skjermtiden"},
            {"tags": ["Plur", "Ind"], "word_form": "skjermtider"},
            {"tags": ["Plur", "Def"], "word_form": "skjermtidene"}]},
        {"tags": ["NOUN", "Fem"], "inflection": [
            {"tags": ["Sing", "Ind"], "word_form": "skjermtid"},
            {"tags": ["Sing", "Def"], "word_form": "skjermtida"},
            {"tags": ["Plur", "Ind"], "word_form": "skjermtider"},
            {"tags": ["Plur", "Def"], "word_form": "skjermtidene"}]}]}]}


def test_parse_uib_noun_fem_priority():
    f = ordbank._parse_uib(_noun_article(), "noun")
    # дублет рода: Fem приоритетнее (конвенция дампа), артикль согласован с def_sg
    assert f == {"pos": "noun", "gender": "ei", "def_sg": "skjermtida",
                 "indef_pl": "skjermtider", "def_pl": "skjermtidene"}


def test_parse_uib_verb():
    art = {"lemmas": [{"lemma": "streame", "paradigm_info": [
        {"tags": ["VERB"], "inflection": [
            {"tags": ["Inf"], "word_form": "streame"},
            {"tags": ["Pres"], "word_form": "streamer"},
            {"tags": ["Past"], "word_form": "streamet"},
            {"tags": ["PerfPart"], "word_form": "streamet"}]}]}]}
    f = ordbank._parse_uib(art, "verb")
    assert f == {"pos": "verb", "present": "streamer", "past": "streamet",
                 "perfect": "har streamet"}


def test_parse_uib_adjective():
    art = {"lemmas": [{"lemma": "kul", "paradigm_info": [
        {"tags": ["ADJ"], "inflection": [
            {"tags": ["Pos", "Masc/Fem"], "word_form": "kul"},
            {"tags": ["Pos", "Neuter"], "word_form": "kult"},
            {"tags": ["Pos", "Plur"], "word_form": "kule"},
            {"tags": ["Cmp"], "word_form": "kulere"},
            {"tags": ["Sup", "Ind"], "word_form": "kulest"}]}]}]}
    f = ordbank._parse_uib(art, "adjective")
    assert f == {"pos": "adjective", "neuter": "kult", "plural": "kule",
                 "comparative": "kulere", "superlative": "kulest"}


def test_parse_uib_none_for_missing():
    assert ordbank._parse_uib({"lemmas": []}, "noun") is None
    assert ordbank._parse_uib(_noun_article(), "adjective") is None
