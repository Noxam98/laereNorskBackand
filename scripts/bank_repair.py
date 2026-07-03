#!/usr/bin/env python3
"""Исполняется НА СЕРВЕРЕ после доставки нового ordbank_new.db (workflow ordbank.yml).

1) Стабильность выбора: если в старом /opt/norsk/data/ordbank.db выбор слота всё ещё
   валиден по variants нового — сохраняем его (дублеты не скачут между версиями дампа:
   krefter не превращается в krafter из-за алфавитного тай-брейка).
2) Ремонт word_pool.forms: значение слота меняется ТОЛЬКО если текущего нет среди
   допустимых вариантов банка; артикль держится согласованным с def_sg; флаги
   uncountable/uninflectable не трогаются.
3) Подмена файла банка + копия в Tigris.
"""
import json
import os
import shutil
import sqlite3

DATA = "/opt/norsk/data"
NEW = f"{DATA}/ordbank_new.db"
CUR = f"{DATA}/ordbank.db"
CHECK = ("gender", "def_sg", "indef_pl", "def_pl", "present", "past", "perfect",
         "neuter", "plural", "comparative", "superlative")


def align_gender(f):
    d = f.get("def_sg", "")
    if d and f.get("pos") == "noun":
        f["gender"] = "ei" if d.endswith("a") else "et" if d.endswith("et") \
            else "en" if d.endswith("en") else f.get("gender")
    return f


def main():
    new = sqlite3.connect(NEW)
    # 1) стабильность против старого банка
    kept = 0
    if os.path.exists(CUR):
        old = sqlite3.connect(f"file:{CUR}?mode=ro", uri=True)
        for no, pos, forms_j in old.execute("SELECT norwegian, pos, forms FROM forms"):
            r = new.execute("SELECT forms, (SELECT v FROM variants WHERE norwegian=? AND pos=?) "
                            "FROM forms WHERE norwegian=? AND pos=?", (no, pos, no, pos)).fetchone()
            if not r:
                continue
            cur_pick, vsets = json.loads(r[0]), json.loads(r[1] or "{}")
            prev = json.loads(forms_j)
            changed = False
            for k, pv in prev.items():
                if k in CHECK and pv in set(vsets.get(k, [])) and cur_pick.get(k) != pv:
                    cur_pick[k] = pv
                    changed = True
            if changed:
                new.execute("UPDATE forms SET forms=? WHERE norwegian=? AND pos=?",
                            (json.dumps(align_gender(cur_pick), ensure_ascii=False), no, pos))
                kept += 1
        new.commit()
        old.close()
    print(f"стабильность: сохранено прежних выборов {kept}")

    # 2) ремонт форм пула: менять только невалидное. Если задан RESTORE_BACKUP
    # (ключ S3 с гзипнутым снапшотом) — значения оттуда приоритетнее текущих
    # (одноразовый откат поспешных правок: валидное старое побеждает).
    old_forms = {}
    bk = os.environ.get("RESTORE_BACKUP")
    if bk:
        import gzip, boto3
        boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"]).download_file(
            os.environ["BUCKET_NAME"], bk, "/tmp/pre.db.gz")
        with gzip.open("/tmp/pre.db.gz", "rb") as fi, open("/tmp/pre.db", "wb") as fo:
            fo.write(fi.read())
        pre = sqlite3.connect("file:/tmp/pre.db?mode=ro", uri=True)
        old_forms = {(w, p): json.loads(f) for w, p, f in pre.execute(
            "SELECT norwegian, pos, forms FROM word_pool WHERE forms IS NOT NULL "
            "AND pos IN ('noun','verb','adjective')")}
        pre.close(); os.remove("/tmp/pre.db"); os.remove("/tmp/pre.db.gz")
        print(f"восстановление из бэкапа {bk}: {len(old_forms)} слов")
    c = sqlite3.connect(f"{DATA}/users.db", timeout=30)
    same = changed = 0
    samples = []
    for wid, no, pos, forms_j in c.execute(
            "SELECT id, norwegian, pos, forms FROM word_pool "
            "WHERE pos IN ('noun','verb','adjective') AND forms IS NOT NULL").fetchall():
        r = new.execute("SELECT forms, (SELECT v FROM variants WHERE norwegian=? AND pos=?) "
                        "FROM forms WHERE norwegian=? AND pos=?",
                        (no.lower(), pos, no.lower(), pos)).fetchone()
        if not r:
            continue
        pick, vsets = json.loads(r[0]), json.loads(r[1] or "{}")
        cur = json.loads(forms_j)
        tgt = dict(cur)
        prev = old_forms.get((no, pos), {})
        for k in CHECK:
            if k not in pick:
                continue
            vs = set(vsets.get(k, []))
            pv, cv = prev.get(k), cur.get(k)
            tgt[k] = pv if pv in vs else (cv if cv in vs else pick[k])
        tgt = align_gender(tgt)
        if tgt == cur:
            same += 1
            continue
        changed += 1
        c.execute("UPDATE word_pool SET forms=? WHERE id=?",
                  (json.dumps(tgt, ensure_ascii=False), wid))
        if len(samples) < 10:
            ch = {k: (cur.get(k), tgt[k]) for k in CHECK if k in tgt and cur.get(k) != tgt.get(k)}
            samples.append(f"{no}/{pos}: " + ", ".join(f"{k} {a}->{b}" for k, (a, b) in ch.items()))
    c.commit()
    print(f"пул: валидных без изменений {same} · исправлено {changed}")
    for s in samples:
        print(" ", s)

    # 3) подмена файла + Tigris
    new.close()
    shutil.move(NEW, CUR)
    shutil.chown(CUR, "norsk", "norsk")
    import boto3
    boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL_S3"]).upload_file(
        CUR, os.environ["BUCKET_NAME"], "ordbank/ordbank.db")
    print("файл банка подменён, Tigris-копия обновлена")


if __name__ == "__main__":
    main()
