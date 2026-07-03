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
    files = {}
    for m in tf.getmembers():
        name = m.name.rsplit("/", 1)[-1]
        if name in ("lemma.txt", "fullformsliste.txt"):
            files[name] = tf.extractfile(m).read().decode("latin-1")
    assert set(files) == {"lemma.txt", "fullformsliste.txt"}, f"в дампе не хватает файлов: {files.keys()}"
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
        n += 1
    out.commit()
    print(f"ordbank.db: {n} парадигм → {dst}")
    for w in ("liv", "skap", "kraft", "potet"):
        r = out.execute("SELECT forms FROM forms WHERE norwegian=? AND pos='noun'", (w,)).fetchone()
        print(f"  {w}: {r[0][:80] if r else 'НЕТ'}")


if __name__ == "__main__":
    main()
