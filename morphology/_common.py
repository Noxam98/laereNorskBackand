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
