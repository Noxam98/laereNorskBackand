"""Нормализация частей речи (норв.↔англ.) + классификация служебных по канону."""
from pos import normalize_pos
from db.learning import is_function_word


def test_normalize_pos_norwegian_to_english():
    assert normalize_pos("substantiv") == "noun"
    assert normalize_pos("adjektiv") == "adjective"
    assert normalize_pos("preposisjon") == "preposition"
    assert normalize_pos("pronomen") == "pronoun"
    assert normalize_pos("konjunksjon") == "conjunction"
    assert normalize_pos("subjunksjon") == "conjunction"
    assert normalize_pos("tallord") == "numeral"
    assert normalize_pos("interjeksjon") == "interjection"
    assert normalize_pos("VERB") == "verb"          # регистр не важен
    assert normalize_pos("noun") == "noun"          # уже англ. — без изменений
    assert normalize_pos("") == "" and normalize_pos(None) == ""


def test_is_function_word_both_spellings():
    # служебное распознаётся И по норв., И по англ. написанию
    assert is_function_word("på", {"part_of_speech": "preposisjon"})
    assert is_function_word("på", {"part_of_speech": "preposition"})   # раньше НЕ распознавалось
    assert is_function_word("og", {"part_of_speech": "konjunksjon"})
    assert is_function_word("og", {"part_of_speech": "conjunction"})
    assert is_function_word("å", {})                                   # частица
    assert is_function_word("ikke", {"part_of_speech": "adverb"})      # функц. наречие (белый список)
    # контентное — не служебное (оба написания)
    assert not is_function_word("bil", {"part_of_speech": "substantiv"})
    assert not is_function_word("bil", {"part_of_speech": "noun"})
