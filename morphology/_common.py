"""Общие морфо-хелперы (нормализация форм, слоги, согласные) для noun/verb/adjective."""
import unicodedata


# ─────────────────────────────────────────────────────────────────────────────
# Общие хелперы
# ─────────────────────────────────────────────────────────────────────────────

VOWELS = set("aeiouyæøå")


def _norm(s):
    """Нормализация для сравнения форм: NFC + strip + lower.

    Критично для ø/å/æ — иначе хранимое 'Bøker' != предсказанное 'bøker'
    даст ложную нерегулярность.
    """
    if s is None:
        return ""
    return unicodedata.normalize("NFC", str(s)).strip().lower()


def _syllables(w):
    """Грубый счётчик слогов = число ГРУПП гласных (aeiouyæøå)."""
    n = 0
    prev_vowel = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_vowel:
            n += 1
        prev_vowel = is_v
    return n


def _ends_double_cons(w):
    """True, если слово кончается на удвоенный согласный (gg, nn, ll, mm, tt...)."""
    return len(w) >= 2 and w[-1] == w[-2] and w[-1] not in VOWELS


def _degeminate_tail(w):
    """Схлопнуть финальный удвоенный согласный: gamm→gam, bygg→byg, kall→kal.

    Используется перед добавлением суффикса, который требует одиночной согласной
    (синкопа прилагательных, упрощение перед -te/-de у глаголов).
    """
    if _ends_double_cons(w):
        return w[:-1]
    return w


def _plausible(f):
    """Отсечь явные малформы (двойная гласная/тройная согласная) — учащийся так не ошибётся.

    Общий фильтр дистракторов для всех частей речи: строка правдоподобна как форма,
    если в ней нет подряд двух одинаковых гласных и трёх одинаковых согласных.
    Плюс мусор-маркеры LLM-заполнения («n/a» у несравнимых форм и т.п.) — они приходят
    соседними формами слова и без фильтра утекали бы в варианты ответа.
    """
    if not f or len(f) < 2:
        return False
    if f.strip().lower() in ("n/a", "na", "none", "null", "ingen") or "/" in f:
        return False
    prev = ""
    run_v = run_c = 0
    for ch in f:
        if ch in VOWELS:
            run_v = run_v + 1 if ch == prev else 1
            run_c = 0
            if run_v >= 2:
                return False
        else:
            run_c = run_c + 1 if ch == prev else 1
            run_v = 0
            if run_c >= 3:
                return False
        prev = ch
    return True
