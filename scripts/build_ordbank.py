#!/usr/bin/env python3
"""Сборка ordbank.db из публичного дампа Norsk ordbank (Språkbanken, CC-BY).

Запуск:  python3 scripts/build_ordbank.py <url-дампа|путь.tar.gz> <выход.db>
Гоняется раннером GitHub Actions (workflow ordbank.yml) — пересборка банка без ssh
с ноута. Таблицы: forms (канонический выбор) + variants (ВСЕ допустимые формы по
слотам — для аудитов «а валидно ли старое значение»).

Правила (выстраданы 3.07.2026, см. память lexin-harvest-pipeline):
  • prop-строки (имена собственные: Liv, Tone) исключаются — травили род;
  • артикль ВСЕГДА согласован с выбранной def_sg (-a→ei, -et→et, -en→en);
  • дублет рода → fem (ei) приоритетнее (конвенция jenta/klokka).
Стабильность выбора между версиями дампа обеспечивает bank_repair.py на сервере.
"""
import collections
import io
import json
import re
import sqlite3
import sys
import tarfile
import urllib.request

CLS = {"subst", "verb", "adj"}
POS = {"subst": "noun", "verb": "verb", "adj": "adjective"}
G = {"mask": "en", "fem": "ei", "nøyt": "et"}


def read_dump(src: str):
    if src.startswith("http"):
        with urllib.request.urlopen(src, timeout=120) as r:
            blob = r.read()
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    else:
        tf = tarfile.open(src, mode="r:gz")
    # leddanalyse.txt — разбор составных слов (sammensetning), опционально: старые дампы без
    # него собираются по-прежнему (просто без таблицы compounds).
    want = {"lemma.txt", "fullformsliste.txt", "leddanalyse.txt"}
    files = {}
    for m in tf.getmembers():
        name = m.name.rsplit("/", 1)[-1]
        if name in want:
            files[name] = tf.extractfile(m).read().decode("latin-1")
    assert {"lemma.txt", "fullformsliste.txt"} <= set(files), f"в дампе не хватает файлов: {sorted(files)}"
    return files


def pick(cands, suffix):
    for c in sorted(cands):
        if c.endswith(suffix):
            return c
    return sorted(cands)[0] if cands else None


def build_noun(rr):
    genders, dsg, ipl, dpl, ubo = set(), set(), set(), set(), set()
    for form, tag in rr:
        m = re.match(r"subst (mask|fem|nøyt) ", tag)
        if " ubøy" in tag:
            if m:
                ubo.add(G[m.group(1)])
            continue
        if not m:
            continue
        genders.add(G[m.group(1)])
        if " ent be" in tag:
            dsg.add(form)
        elif " fl ub" in tag:
            ipl.add(form)
        elif " fl be" in tag:
            dpl.add(form)
    if not genders:
        if ubo:
            g = "ei" if "ei" in ubo else ("en" if "en" in ubo else "et")
            return {"pos": "noun", "gender": g, "uninflectable": True}
        return None
    g = "ei" if "ei" in genders else ("en" if "en" in genders else "et")
    out = {"pos": "noun", "gender": g}
    if dsg:
        out["def_sg"] = pick(dsg, "a" if g == "ei" else ("et" if g == "et" else "en"))
        d = out["def_sg"]
        out["gender"] = "ei" if d.endswith("a") else "et" if d.endswith("et") \
            else "en" if d.endswith("en") else g
    if ipl:
        out["indef_pl"] = pick(ipl, "er") or sorted(ipl)[0]
    if dpl:
        out["def_pl"] = pick(dpl, "ene")
    return out


def build_verb(rr):
    pres, pret, perf = set(), set(), set()
    for form, tag in rr:
        if " pass" in tag:
            continue
        if " pres" in tag:
            pres.add(form)
        elif " pret" in tag:
            pret.add(form)
        elif "perf-part" in tag:
            perf.add(form)
    out = {"pos": "verb"}
    if pres:
        out["present"] = pick(pres, "er") or sorted(pres)[0]
    if pret:
        out["past"] = pick(pret, "te") or sorted(pret)[0]
    if perf:
        out["perfect"] = "har " + (pick(perf, "t") or sorted(perf)[0])
    return out if len(out) > 1 else None


def build_adj(rr):
    neu, plu, komp, sup = set(), set(), set(), set()
    for form, tag in rr:
        if "pos nøyt ub ent" in tag:
            neu.add(form)
        elif re.search(r"pos .*fl", tag):
            plu.add(form)
        elif " komp" in tag:
            komp.add(form)
        elif " sup ub" in tag:
            sup.add(form)
    out = {"pos": "adjective"}
    if neu:
        out["neuter"] = pick(neu, "t")
    if plu:
        out["plural"] = pick(plu, "e")
    if komp:
        out["comparative"] = pick(komp, "ere")
    if sup:
        out["superlative"] = pick(sup, "st")
    return out if len(out) > 1 else None


def variant_sets(cls, rr):
    vs = collections.defaultdict(set)
    for form, tag in rr:
        if cls == "subst":
            m = re.match(r"subst (mask|fem|nøyt) ", tag)
            if m:
                vs["gender"].add(G[m.group(1)])
            if " ent be" in tag:
                vs["def_sg"].add(form)
            elif " fl ub" in tag:
                vs["indef_pl"].add(form)
            elif " fl be" in tag:
                vs["def_pl"].add(form)
        elif cls == "verb":
            if " pass" in tag:
                continue
            if " pres" in tag:
                vs["present"].add(form)
            elif " pret" in tag:
                vs["past"].add(form)
            elif "perf-part" in tag:
                vs["perfect"].add("har " + form)
        else:
            if "pos nøyt ub ent" in tag:
                vs["neuter"].add(form)
            elif re.search(r"pos .*fl", tag):
                vs["plural"].add(form)
            elif " komp" in tag:
                vs["comparative"].add(form)
            elif " sup ub" in tag:
                vs["superlative"].add(form)
    return {k: sorted(v) for k, v in vs.items()}


# ── составные слова (sammensetning): leddanalyse.txt даёт ГОТОВЫЙ авторитетный разбор ──
# Колонки (tab): 2=OPPSLAG, 4=FORLEDD, 6=FUGE, 7=ETTERLEDD, 11=OPPSLAG_LEDD_MARKERT.
# forledd (первый элемент, лемма) + fuge (соединитель -s-/-e-/пусто) + etterledd (голова).
# Разбор бинарный и рекурсивный: barnehagelærer=barnehage+lærer, а barnehage=barn+e+hage —
# части сами леммы, кликаются в карточке. Части в поле FORLEDD иногда с дефисом шва (kjøle-) →
# срезаем. Пропускаем: simplex (не композит, пустые части — universitet/telefon), имена
# собственные (капитал → мусор Andalucia=Anda+lucia) и не-словарные ключи (1. mai-tog,
# 14C-datering, A-avis). Строки задублированы по роду → PRIMARY KEY гасит (первый выигрывает).
def build_compounds(out, text):
    for line in text.splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 12:
            continue
        oppslag = p[2].strip()
        forledd, fuge, etterledd = p[4].strip().strip("-"), p[6].strip(), p[7].strip().strip("-")
        marked = p[11].strip()
        if not forledd or not etterledd:                     # simplex — не составное
            continue
        key = oppslag.lower()
        if not key.isalpha() or not oppslag[:1].islower():   # имя собственное / не-слово — мимо
            continue
        out.execute("INSERT OR IGNORE INTO compounds VALUES (?,?,?,?,?)",
                    (key, forledd, fuge, etterledd, marked))
    return out.execute("SELECT COUNT(*) FROM compounds").fetchone()[0]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    files = read_dump(src)
    lemma_of = {}
    for line in files["lemma.txt"].splitlines()[1:]:
        p = line.rstrip("\r").split("\t")
        if len(p) >= 3:
            lemma_of[p[1]] = p[2].strip().lower()
    rows = collections.defaultdict(list)
    for line in files["fullformsliste.txt"].splitlines()[1:]:
        p = line.split("\t")
        if len(p) < 4 or p[1] not in lemma_of or "normert" not in p[3] or " prop" in p[3]:
            continue
        cls = p[3].split()[0] if p[3].strip() else ""
        if cls in CLS:
            rows[(lemma_of[p[1]], cls)].append((p[2].strip().lower(), p[3]))
    out = sqlite3.connect(dst)
    out.execute("CREATE TABLE forms (norwegian TEXT NOT NULL, pos TEXT NOT NULL, forms TEXT NOT NULL, "
                "PRIMARY KEY (norwegian, pos)) WITHOUT ROWID")
    out.execute("CREATE TABLE variants (norwegian TEXT NOT NULL, pos TEXT NOT NULL, v TEXT NOT NULL, "
                "PRIMARY KEY (norwegian, pos)) WITHOUT ROWID")
    # обратный индекс словоформ: gikk → gå/verb (поиск по любой форме)
    out.execute("CREATE TABLE formindex (form TEXT NOT NULL, norwegian TEXT NOT NULL, pos TEXT NOT NULL, "
                "PRIMARY KEY (form, norwegian, pos)) WITHOUT ROWID")
    # разбор составных слов (leddanalyse) — forledd (+fuge) + etterledd; marked = «barne-hage»
    out.execute("CREATE TABLE compounds (norwegian TEXT PRIMARY KEY, forledd TEXT NOT NULL, "
                "fuge TEXT, etterledd TEXT NOT NULL, marked TEXT) WITHOUT ROWID")
    build = {"subst": build_noun, "verb": build_verb, "adj": build_adj}
    n = 0
    for (lemma, cls), rr in rows.items():
        f = build[cls](rr)
        if not f:
            continue
        out.execute("INSERT OR REPLACE INTO forms VALUES (?,?,?)",
                    (lemma, POS[cls], json.dumps(f, ensure_ascii=False)))
        out.execute("INSERT OR REPLACE INTO variants VALUES (?,?,?)",
                    (lemma, POS[cls], json.dumps(variant_sets(cls, rr), ensure_ascii=False)))
        out.executemany("INSERT OR IGNORE INTO formindex VALUES (?,?,?)",
                        [(form, lemma, POS[cls]) for form, _tag in rr])
        n += 1
    nc = build_compounds(out, files["leddanalyse.txt"]) if "leddanalyse.txt" in files else 0
    out.commit()
    print(f"ordbank.db: {n} парадигм, {nc} составных слов → {dst}")
    for w in ("liv", "skap", "kraft", "potet"):
        r = out.execute("SELECT forms FROM forms WHERE norwegian=? AND pos='noun'", (w,)).fetchone()
        print(f"  {w}: {r[0][:80] if r else 'НЕТ'}")
    for w in ("barnehage", "arbeidsplass", "barnehagelærer", "universitet"):
        r = out.execute("SELECT forledd, fuge, etterledd FROM compounds WHERE norwegian=?", (w,)).fetchone()
        print(f"  сост. {w}: {r if r else 'simplex/нет'}")


if __name__ == "__main__":
    main()
