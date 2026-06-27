"""Чистые правила онлайн-игры (без сокетов и без состояния комнаты): подсчёт очков за ответ,
нормализация печатного ответа, валидация/кламп настроек комнаты, формирование payload вопроса
и выбор правильного ответа/ключей по направлению.

Зависит только от stdlib — поэтому юнит-тестируемо и переиспользуемо. online.py реэкспортирует
это обратно (диспетчер игр GAMES и сам WebSocket-поток остаются там).
"""
import math

# --- Границы/дефолты настроек комнаты (числа — здесь: их кламп есть чистая логика) ---
QUESTION_TIME = 15      # дефолт сек на вопрос (если не задано в настройках комнаты)
MIN_PLAYERS = 2
MAX_PLAYERS_CAP = 8
COUNT_MIN, COUNT_MAX = 3, 20
QTIME_MIN, QTIME_MAX = 5, 30   # границы длительности вопроса
GAME_KEYS = ("quiz", "race")   # допустимые типы игр; реестр-диспетчер GAMES (runner'ы) живёт в online.py


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _norm_settings(s):
    s = s or {}
    return {
        "game": s.get("game") if s.get("game") in GAME_KEYS else "quiz",
        "answer": "choice" if s.get("answer") == "choice" else "type",  # гонка: печать / выбор из 4
        "dir": "int2no" if s.get("dir") == "int2no" else "no2int",
        "source": s.get("source") if s.get("source") in ("pool", "dict", "ai") else "pool",  # pool / словари хоста / AI-подбор
        "dictId": int(s["dictId"]) if str(s.get("dictId") or "").isdigit() else None,  # None=все словари хоста
        "level": (s.get("level") or "") or None,     # A1..C2 или None=любой
        "topic": (s.get("topic") or "") or None,     # ключ темы или None=любая
        "count": _clamp(s.get("count"), COUNT_MIN, COUNT_MAX, 7),
        "qtime": _clamp(s.get("qtime"), QTIME_MIN, QTIME_MAX, QUESTION_TIME),
        "maxPlayers": _clamp(s.get("maxPlayers"), MIN_PLAYERS, MAX_PLAYERS_CAP, 4),
        "private": bool(s.get("private")),
    }


def _q_payload(q, i, total, lang, qtime):
    if q["per_lang"]:   # no2int — показываем норвежское слово, варианты-переводы
        return {"type": "question", "i": i, "total": total, "dir": "no2int",
                "prompt": q["no"], "options": q["options"][lang], "keys": q["keys"][lang], "time": qtime}
    return {"type": "question", "i": i, "total": total, "dir": "int2no",   # перевод → норв. слова
            "prompt": q["prompt"][lang], "options": q["options"], "keys": q["keys"], "time": qtime}


def _q_correct(q, lang):
    return q["correct"][lang] if q["per_lang"] else q["correct"]


def _q_keys(q, lang):
    return q["keys"][lang] if q["per_lang"] else q["keys"]


def _gain(elapsed, correct, streak):
    """Очки за верный ответ: резкая зависимость от скорости (экспонента) + бонус за серию.
    ~880 при мгновенном ответе, ~390 к 4 сек, ~190 к 8 сек; серия даёт до +250."""
    if not correct:
        return 0
    speed = round(900 * math.exp(-elapsed / 3.5))
    bonus = min(max(streak - 1, 0), 5) * 50
    return 100 + speed + bonus


def _norm_answer(s):
    """Нормализация печатного ответа для сравнения: регистр, пробелы, артикли å/en/ei/et."""
    s = (s or "").strip().lower()
    s = " ".join(s.split())
    for art in ("å ", "en ", "ei ", "et "):
        if s.startswith(art):
            s = s[len(art):]
            break
    return s
