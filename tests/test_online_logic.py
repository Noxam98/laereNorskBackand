"""Чистые правила онлайн-игры (без сокетов): очки за ответ, нормализация печатного ответа,
валидация/кламп настроек комнаты, выбор правильного варианта/ключей по направлению."""
import online


def test_gain_scoring_curve():
    assert online._gain(0.1, False, 0) == 0                     # неверно → 0 очков
    fast = online._gain(0.1, True, 1)
    slow = online._gain(8.0, True, 1)
    assert fast > slow > 100                                    # быстрее → больше очков, всегда > базы
    assert online._gain(0.1, True, 6) > online._gain(0.1, True, 1)   # серия даёт бонус


def test_norm_answer_strips_articles_and_case():
    assert online._norm_answer("  En  Hund ") == "hund"
    assert online._norm_answer("å spise") == "spise"
    assert online._norm_answer("HUS") == "hus"
    assert online._norm_answer("ei jente") == "jente"


def test_norm_settings_validates_and_clamps():
    s = online._norm_settings({"game": "что-то", "count": 999, "qtime": 1, "maxPlayers": "abc"})
    assert s["game"] == "quiz"                                  # неизвестная игра → quiz
    assert s["count"] != 999                                    # вне диапазона → клампится
    assert s["maxPlayers"] == 4                                 # нечисло → дефолт
    assert s["dir"] == "no2int" and s["answer"] == "type"       # дефолты направления/ответа
    assert online._norm_settings(None)["game"] == "quiz"        # None → дефолтные настройки


def test_q_correct_and_keys_by_direction():
    per_lang = {"per_lang": True, "correct": {"ru": "собака"}, "keys": {"ru": ["собака"]}}
    assert online._q_correct(per_lang, "ru") == "собака"
    assert online._q_keys(per_lang, "ru") == ["собака"]
    shared = {"per_lang": False, "correct": "hund", "keys": ["hund"]}
    assert online._q_correct(shared, "ru") == "hund"
    assert online._q_keys(shared, "ru") == ["hund"]
