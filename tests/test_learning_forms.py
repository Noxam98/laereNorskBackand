"""Тесты трека форм (чистая логика learning_forms: клетки, диспетч опций, элементы, планировщик)."""
import json
from db.learning_forms import (
    FORM_STAGES, form_cells_for, form_options, form_element, cell_value, schedule_form,
)


def _data(pos):
    return json.dumps({"part_of_speech": pos, "translate": {"ru": "x"}})


def test_form_cells_for():
    # сущ.: только присутствующие формы
    n = {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}
    assert set(form_cells_for("noun", n)) == {"gender", "indef_pl", "def_sg", "def_pl"}
    assert form_cells_for("noun", {"gender": "en"}) == ["gender"]        # пустые формы пропущены
    # глаг.: present/past/perfect
    v = {"present": "går", "past": "gikk", "perfect": "har gått"}
    assert set(form_cells_for("verb", v)) == {"present", "past", "perfect"}
    # прил.: перифраз 'mer praktisk' — не одна словоформа → мимо
    a = {"neuter": "praktisk", "plural": "praktiske", "comparative": "mer praktisk", "superlative": "mest praktisk"}
    assert set(form_cells_for("adjective", a)) == {"neuter", "plural"}
    # местоимение — морфологии форм нет
    assert form_cells_for("pronoun", {"obj": "meg"}) == []


def test_cell_value_perfect_participle():
    # перфект глагола дрилим как причастие (без 'har')
    assert cell_value("verb", {"perfect": "har gått"}, "perfect") == "gått"
    assert cell_value("noun", {"gender": "et"}, "gender") == "et"


def test_form_options_dispatch():
    # сущ. gender → 2 других артикля
    c, d = form_options("noun", "bil", {"gender": "en", "indef_pl": "biler", "def_sg": "bilen", "def_pl": "bilene"}, "gender")
    assert c == "en" and set(d) == {"ei", "et"}
    # глаг. сильный: наивная слабая форма среди дистракторов
    c, d = form_options("verb", "gå", {"present": "går", "past": "gikk", "perfect": "har gått"}, "past")
    assert c == "gikk" and "gådde" in d
    # прил.: классическая ошибка согласования
    c, d = form_options("adjective", "grønn", {"neuter": "grønt", "plural": "grønne", "comparative": "grønnere", "superlative": "grønnest"}, "neuter")
    assert c == "grønt" and "grønnt" in d


def test_form_element_stages():
    row = {"pool_id": 7, "norwegian": "gå", "mastered": 1}
    forms = {"present": "går", "past": "gikk", "perfect": "har gått"}
    data = _data("verb")
    # card — пассивный показ формы
    card = form_element(row, forms, data, "past", "card")
    assert card["mode"] == "study" and card["reveal"] == "gikk" and card["form_track"] is True
    # choose — варианты содержат correct + дистракторы, correct первым в target
    ch = form_element(row, forms, data, "past", "choose")
    ws = [o["w"] for o in ch["options"]]
    assert ch["mode"] == "choice" and ch["target"]["value"] == "gikk" and "gikk" in ws and "gådde" in ws
    assert ch["target"]["value"] not in ch["distractors"]
    # produce — ввод формы
    pr = form_element(row, forms, data, "past", "produce")
    assert pr["mode"] == "input" and pr["target"]["value"] == "gikk"


def test_schedule_form_ramp():
    # card → choose, ещё в этой сессии
    assert schedule_form("card", 2.5, 0, True)[0] == "choose"
    assert schedule_form("card", 2.5, 0, True)[3] == 0
    # choose верно → produce, всё ещё в сессии (due 0)
    ns, ease, iv, due = schedule_form("choose", 2.5, 0, True)
    assert ns == "produce" and due == 0 and ease > 2.5
    # produce верно → клетка отработана, планируем повтор (due ≥ 1 день)
    ns, ease, iv, due = schedule_form("produce", 2.5, 0, True)
    assert ns == "produce" and iv >= 1 and due >= 1
    # produce ошибка → шаг назад к choose, ease вниз, повтор в сессии
    ns, ease, iv, due = schedule_form("produce", 2.5, 5, False)
    assert ns == "choose" and ease < 2.5 and due == 0
    # choose ошибка → назад к card
    assert schedule_form("choose", 2.5, 0, False)[0] == "card"
    # интервал растёт мультипликативно и упирается в потолок
    _, _, iv, _ = schedule_form("produce", 3.0, 200, True)
    assert iv <= 365


def test_form_stages_order():
    assert FORM_STAGES == ("card", "choose", "produce")
