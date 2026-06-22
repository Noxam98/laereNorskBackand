"""Нечёткая сверка введённого ответа с допустимыми формами (для экзамена с вводом текста).
Чистый Python, без зависимостей. Принцип: НОРМАЛИЗАЦИЯ → расстояние Дамерау-Левенштейна (OSA,
учитывает перестановку соседних букв) с порогом, масштабируемым по длине. Фразы — потокенно.

Сравниваем ТОЛЬКО со «своими» формами слова (переводы/норвежское), не с дистракторами, —
поэтому «близкий» промах = опечатка верного ответа, а не принятие чужого слова."""
import re
import unicodedata

_FOLD = str.maketrans({"å": "a", "ø": "o", "æ": "ae"})
# короткие служебные/артикли — выкидываем при потокенном сравнении фраз
_GLUE = {"en", "ei", "et", "å", "a", "an", "the", "to", "of", "i", "på"}


def normalize(s):
    """Нижний регистр, å/ø/æ→a/o/ae, снятие диакритики (é→e, ё→е), пунктуация→пробел, схлоп пробелов."""
    s = (s or "").strip().lower().translate(_FOLD)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def osa(a, b):
    """Расстояние Дамерау-Левенштейна (Optimal String Alignment): вставка/удаление/замена +
    перестановка соседних — каждое по 1. Быстрый отсев при сильной разнице длин."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 3:
        return 99
    if la == 0 or lb == 0:
        return la or lb
    prev2 = None
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[lb]


def tol_for(n):
    """Сколько правок прощаем для слова длины n: ≤3 — 0 (точно), 4–7 — 1, 8+ — 2."""
    return 0 if n <= 3 else 1 if n <= 7 else 2


def word_close(a, b):
    """Близко ли одиночное слово a к эталону b (оба уже нормализованы)."""
    if a == b:
        return True
    return osa(a, b) <= tol_for(len(b))


def _tokens(s):
    return [w for w in s.split() if w not in _GLUE] or s.split()


def phrase_close(a, b):
    """Фраза a близка к эталону b: каждый значимый токен эталона покрыт каким-то токеном ввода
    (в пределах порога, порядок неважен; лишние слова во вводе допускаем)."""
    at, bt = _tokens(a), _tokens(b)
    if not bt:
        return False
    used = [False] * len(at)
    for tok in bt:
        hit = False
        for i, w in enumerate(at):
            if not used[i] and word_close(tok, w):
                used[i] = True
                hit = True
                break
        if not hit:
            return False
    return True


def word_forms(norwegian, forms):
    """Все поверхностные формы слова для приёма ответа: лемма + словоформы из колонки forms
    (сущ.: def_sg/indef_pl/def_pl; глаг.: present/past/perfect; прил.: neuter/plural/...).
    Исключаем служебные ключи pos/gender. Уникальные, с сохранением порядка."""
    out = [norwegian] if norwegian else []
    if isinstance(forms, dict):
        for k, v in forms.items():
            if k in ("pos", "gender"):
                continue
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    seen, res = set(), []
    for w in out:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            res.append(w)
    return res


def fuzzy_match(inp, forms, known=None):
    """True, если ввод inp достаточно близок хотя бы к одной из допустимых форм forms.
    known — множество УЖЕ нормализованных «известных слов» (словарь). Если задано и ввод —
    это известное ДРУГОЕ слово (не одна из forms), считаем это осознанным неверным ответом
    (а не опечаткой) и отклоняем. Так edit-distance не принимает соседей вроде hund/hånd."""
    a = normalize(inp)
    if not a:
        return False
    norm_forms = [normalize(f) for f in (forms or []) if f]
    if a in norm_forms:               # точное совпадение после нормализации — всегда принимаем
        return True
    if known and (" " not in a) and (a in known):   # ввод — известное другое слово → не опечатка
        return False
    for b in norm_forms:
        if (" " in b) or (" " in a):
            if phrase_close(a, b):
                return True
        elif word_close(a, b):
            return True
    return False
