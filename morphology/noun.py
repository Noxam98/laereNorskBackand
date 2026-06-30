"""Морфология существительных (bokmål): род, склонение, мн.ч., детектор нерегулярности, дистракторы."""
from ._common import _norm, _degeminate_tail



# Умлаут-/супплетив-мн. — НЕ выводим правилом, только детектим как нерегулярные.
UMLAUT_NOUNS = {
    "bok", "rot", "bot", "fot", "natt", "hånd", "stang", "tann", "kraft",
    "strand", "and", "tå", "klo", "ku", "gås", "mann", "bonde",
    "kne", "tre", "øye", "stad",
}

# Супплетивы родства (основа меняется + умлаут): far→fedre, mor→mødre ...
SUPPLETIVE_NOUNS = {"far", "mor", "bror", "datter", "søster"}

# Нулевое мн. среднего рода: stored_indef_pl == lemma считается РЕГУЛЯРНЫМ (класс),
# не дрилим как умлаут. Держим явным списком (по слогам ненадёжно).
ZERO_PLURAL_NEUTER = {
    "hus", "barn", "år", "dyr", "egg", "ord", "fjell", "glass", "lys",
    "tak", "fag", "kort", "land", "rom", "slag", "skritt", "kinn", "fjes",
    "kjøkken", "våpen", "tegn", "navn",
}

# Синкопа -el/-en/-er: убрать безударное -e- перед -er (+ дегеминация).
# Эвристика по суффиксу ненадёжна (våpen→våpen, lærer→lærere), держим список.
SYNCOPE_NOUNS = {
    "sykkel", "nøkkel", "regel", "seddel", "vinter", "sommer", "finger",
    "teater", "mønster", "vær", "alder", "åker", "fanger",
    "muskel", "tittel", "artikkel", "onkel", "tabell",  # часть с дегеминацией
}

# Агентивы на -er: мн. -ere (НЕ синкопа). lærer→lærere, arbeider→arbeidere.
AGENTIVE_ER = {
    "lærer", "arbeider", "baker", "tysker", "danser", "spiller", "leser",
    "forsker", "maler", "kelner", "borger", "innbygger", "utlending",
    "fisker", "jeger",
}

# Латинизмы: основа усекается (museum→muse+er).
LATIN_NOUNS = {
    "museum", "faktum", "sentrum", "tema", "drama", "studium", "stadium",
    "gymnasium", "akvarium",
}

# Pluralia tantum / без ед.ч. → нет def_sg, флаг no_singular.
PLURALIA_TANTUM = {
    "foreldre", "briller", "penger", "klær", "omgivelser", "møbler",
}


def _is_zero_plural_neuter(no, gender):
    return gender == "et" and _norm(no) in ZERO_PLURAL_NEUTER


def regular_indef_pl(no, gender):
    """Неопределённое мн.ч.; None если не уверены (умлаут/латинизм/нулевое).

    База bokmål: основа + -er. Спецветки:
      - агентив -er → +e (lærer→lærere);
      - синкопа -el/-en/-er → дегеминация + -er (sykkel→sykler);
      - нулевое мн. среднего (явный список) → = лемме;
      - лемма на -e → +r (jente→jenter);
      - иначе → +er (bil→biler, gutt→gutter).
    """
    l = _norm(no)
    if not l:
        return None

    # Слова, которые правилом НЕ выводим (детектор поймает по расхождению):
    if l in UMLAUT_NOUNS or l in SUPPLETIVE_NOUNS or l in LATIN_NOUNS:
        return None

    # Нулевое мн. среднего рода — регулярно классом, форма = лемме.
    if _is_zero_plural_neuter(no, gender):
        return l

    # Агентив на -er: +e (НЕ синкопа).
    if l in AGENTIVE_ER:
        return l + "e"

    # Синкопа -el/-en/-er: убрать -e-, дегеминировать, + er.
    if l in SYNCOPE_NOUNS and len(l) > 2 and l[-2] == "e":
        base = _degeminate_tail(l[:-2])  # 'sykk' от 'sykkel' → 'syk' ... но sykk→syk?
        # NB: sykkel→sykler требует syk+l? нет: основа 'sykk', cons 'l' → syk+l? =sykl
        # Корректно: sykkel→sykl+er. Дегеминируем 'sykk'→'syk', добавляем 'l'.
        return base + l[-1] + "er"

    # Лемма на -e: +r.
    if l.endswith("e"):
        return l + "r"

    # По умолчанию: +er.
    return l + "er"


def regular_def_sg(no, gender):
    """Опр. ед.ч. (постпозитивный артикль), зависит от рода. None если не уверены."""
    l = _norm(no)
    if not l:
        return None
    if l in PLURALIA_TANTUM:
        return None  # нет ед.ч.
    if l in LATIN_NOUNS or l in SUPPLETIVE_NOUNS:
        return None  # museum→museet и т.п. правилом не строим

    ends_e = l.endswith("e")
    if gender == "en":
        return l + ("n" if ends_e else "en")        # bil→bilen, gutt→gutten
    if gender == "ei":
        return (l[:-1] + "a") if ends_e else (l + "a")  # bok→boka, jente→jenta
    if gender == "et":
        return l + ("t" if ends_e else "et")        # hus→huset, eple→eplet
    # неизвестный род — дефолт мужской
    return l + ("n" if ends_e else "en")


def regular_def_pl(no, gender, indef_pl=None):
    """Опр. мн.ч. Возвращает ОДНУ каноничную строку (для дрилла/дистракторов).

    Правило: от мн.ч. на -er → основа+ene (biler→bilene, sykler→syklene).
    Для нулевого среднего → каноник husene (но детектор примет и husa).
    None если indef_pl неизвестен.
    """
    ip = _norm(indef_pl) if indef_pl is not None else regular_indef_pl(no, gender)
    if ip is None:
        return None
    l = _norm(no)

    # Агентив lærere → lærerne (мн. -ere; опр. теряет -e- перед -ne).
    if l in AGENTIVE_ER and ip.endswith("ere"):
        return ip[:-1] + "ne"    # lærere→lærer+ne=lærerne
    # Форма на -er (включая синкопические sykler):
    if ip.endswith("er"):
        return ip[:-2] + "ene"   # biler→bilene, jenter→jentene, sykler→syklene
    if ip.endswith("e"):         # прочие формы на -e
        return ip + "ne"
    if ip == l:                  # нулевое мн. среднего: каноник -ene
        return ip + "ene"        # hus→husene (детектор примет и husa)
    if ip.endswith("r"):         # форма на -r (на всякий случай)
        return ip + "ne"
    return ip + "ene"


def _allowed_indef_pl(no, gender):
    """Множество ДОПУСТИМЫХ регулярных мн.ч. (с дублетами) для детектора."""
    l = _norm(no)
    out = set()
    pred = regular_indef_pl(no, gender)
    if pred:
        out.add(pred)
    # дублеты синкопических: vinter→vintrer/vintre, sommer→somrer/somre
    if l in SYNCOPE_NOUNS and len(l) > 2 and l[-2] == "e":
        base = _degeminate_tail(l[:-2]) + l[-1]
        out.add(base + "er")
        out.add(base + "e")
    # нулевое среднее: и -er тоже изредка встречается, но каноник = лемма
    if _is_zero_plural_neuter(no, gender):
        out.add(l)
    return out


def _allowed_def_pl(no, gender, indef_pl=None):
    """Множество допустимых опр. мн.ч. (учёт дублетов a/ene)."""
    out = set()
    canon = regular_def_pl(no, gender, indef_pl)
    if canon:
        out.add(canon)
    ip = _norm(indef_pl) if indef_pl is not None else regular_indef_pl(no, gender)
    l = _norm(no)
    # дублеты синкопических def_pl
    if l in SYNCOPE_NOUNS and len(l) > 2 and l[-2] == "e":
        base = _degeminate_tail(l[:-2]) + l[-1]
        out.add(base + "ene")        # vintrene
    # нулевое среднее: husene И husa, barna И barnene
    if _is_zero_plural_neuter(no, gender):
        out.add(l + "ene")           # husene
        out.add(l + "a")             # husa / barna
        # синкопа в опр.мн. у основ на -el/-en/-er: våpen→våpnene, kjøkken→kjøknene
        if len(l) > 2 and l[-2] == "e" and l[-1] in "lnr":
            base = _degeminate_tail(l[:-2]) + l[-1]
            out.add(base + "ene")    # våpnene
            out.add(base + "a")      # våpna
    return out


def is_irregular_noun(no, gender, forms):
    """Детектор нерегулярности сущ. forms: {'def_sg','indef_pl','def_pl'}.

    Возвращает (bool is_irregular, str reason).
    Политика: при расхождении/неуверенности → True.
    """
    l = _norm(no)

    # 0) Явные списки сильнее сравнения.
    if l in PLURALIA_TANTUM:
        return True, "pluralia_tantum"   # нет ед.ч., формы особые
    if l in UMLAUT_NOUNS or l in SUPPLETIVE_NOUNS or l in LATIN_NOUNS:
        return True, "explicit_list"

    stored_ip = _norm(forms.get("indef_pl"))
    stored_dsg = _norm(forms.get("def_sg"))
    stored_dpl = _norm(forms.get("def_pl"))

    # 1) Главный сигнал — мн.ч.
    if stored_ip:
        allowed_ip = _allowed_indef_pl(no, gender)
        if not allowed_ip:
            return True, "indef_pl_unpredictable"
        if stored_ip not in allowed_ip:
            # нулевое мн. там, где ждём -er, и слова нет в белом списке → нерег.
            if stored_ip == l and not _is_zero_plural_neuter(no, gender):
                return True, "unexpected_zero_plural"
            return True, "indef_pl_mismatch"
    else:
        # нет хранимого мн. — при неуверенности склоняемся к True
        if regular_indef_pl(no, gender) is None:
            return True, "no_indef_pl_assume_irregular"

    # 2) Опр. ед.ч. (учёт рода).
    if stored_dsg:
        pred_dsg = regular_def_sg(no, gender)
        if pred_dsg is not None and stored_dsg != pred_dsg:
            return True, "def_sg_mismatch"

    # 3) Опр. мн.ч. (учёт дублетов).
    if stored_dpl:
        allowed_dpl = _allowed_def_pl(no, gender, stored_ip or None)
        if allowed_dpl and stored_dpl not in allowed_dpl:
            return True, "def_pl_mismatch"

    return False, "regular"


# ── Дистракторы для существительного ────────────────────────────────────────

def _competing_def_sg(no, gender):
    """Артикль ЧУЖОГО рода (для дистрактора def_sg)."""
    l = _norm(no)
    ends_e = l.endswith("e")
    out = []
    forms_by_gender = {
        "en": l + ("n" if ends_e else "en"),
        "ei": (l[:-1] + "a") if ends_e else (l + "a"),
        "et": l + ("t" if ends_e else "et"),
    }
    correct = forms_by_gender.get(gender)
    for g, f in forms_by_gender.items():
        if g != gender and f != correct:
            out.append(f)
    return out


def distractors_noun(no, gender, forms):
    """Правдоподобные неверные формы сущ. для клеток выбора.

    Возвращает dict {form_name: [distractors]} без правильных значений.
    """
    l = _norm(no)
    res = {"indef_pl": [], "def_sg": [], "def_pl": []}

    correct_ip = _norm(forms.get("indef_pl"))
    correct_dsg = _norm(forms.get("def_sg"))
    correct_dpl = _norm(forms.get("def_pl"))

    # indef_pl: наивное регулярное там, где слово нерегулярно (bok→*boker, mann→*manner)
    naive_ip = set()
    if l.endswith("e"):
        naive_ip.add(l + "r")
    naive_ip.add(l + "er")           # *boker, *manner, *foter
    if l in AGENTIVE_ER:
        # для агентива: ложная синкопа (lærer→*lærre) и чужой класс (=лемме)
        if len(l) > 2 and l[-2] == "e":
            naive_ip.add(_degeminate_tail(l[:-2]) + l[-1] + "er")  # *lærrer-ish
        naive_ip.add(l)              # *lærer (=лемме)
    if l in SYNCOPE_NOUNS:
        naive_ip.add(l + "er")       # НЕ применить синкопу: sykkel→*sykkeler
    res["indef_pl"] = [f for f in naive_ip if f and f != correct_ip]

    # def_sg: чужой род
    res["def_sg"] = [f for f in _competing_def_sg(no, gender) if f != correct_dsg]
    # двойная гласная на -e лемме (eple→*epleet)
    if l.endswith("e"):
        if gender == "et":
            dbl = l + "et"
        elif gender == "ei":
            dbl = l + "a"
        else:
            dbl = l + "en"
        if dbl != correct_dsg and dbl not in res["def_sg"]:
            res["def_sg"].append(dbl)

    # def_pl: чужой суффикс (-ene↔-a, лишнее -er+ne)
    naive_dpl = set()
    if correct_ip:
        naive_dpl.add(correct_ip + "ne")     # *bilerne, *jenterne
        naive_dpl.add(correct_ip + "e")
    naive_dpl.add(l + "er")                   # *barner
    naive_dpl.add(l + "a")                    # *bila
    res["def_pl"] = [f for f in naive_dpl if f and f != correct_dpl]

    return res
