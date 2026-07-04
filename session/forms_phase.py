"""Фаза форм дневной сессии: чистая FSM цикла «слова ↔ формы» (Этап 5).

Две функции вместо одной из-за IO-границы: вселенная клеток (build_universe)
определяет, ЧЬИ SRS-состояния грузить из form_srs, поэтому оркестратор
(db/learning.build_session) зовёт build_universe → грузит fstates и form_cycle →
plan_forms_phase. Обе функции чистые: БД, настройки и морфология приходят
данными и замыканиями.

Контракт результата (FormsPlan-dict): cap_new/cards_cap — ПОЛЯ результата,
а не мутации входа (перепрофилирование квоты новых слов внутри блока форм
больше не может «случайно» утечь в другие ветки сессии); решение о записи
цикла — save_cycle=(phase, batch) — персистит оркестратор.
"""
import json


def build_universe(enriched, *, ordered_pids, group_of, pronoun_forms, cells_for):
    """Кандидаты грамм-тира и вселенная клеток трека форм.

    Возвращает (cands, info): cands=[(e, group, fdict)] — выученные слова
    грамм-формируемых групп с формами (включая местоим./притяж. с курируемой
    парадигмой); info={pid: {e, fdict, cells}} — только трек форм
    (noun/verb/adjective) со СВОИМИ реальными клетками.

    Слова, уже взятые base-сессией (ordered_pids), грамматикой НЕ резервируем:
    иначе слот квоты ушёл бы на слово, которое отсеется дедупом сессии.
    group_of(e) → группа или None (POS + пер-POS тумблеры настроек);
    pronoun_forms(no) → курируемая парадигма; cells_for(e, fdict) → клетки.
    """
    cands = []
    for e in enriched:
        if e["status"] != "mastered" or e["row"]["pool_id"] in ordered_pids:
            continue
        group = group_of(e)
        if not group:
            continue
        raw = e["row"].get("forms")
        if raw:                                   # сущ./глаг./прил.: формы из БД
            try:
                fdict = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
        elif group == "pronoun":                  # форм в БД нет → курируемая парадигма
            fdict = pronoun_forms((e["row"]["norwegian"] or "").strip().lower())
        else:
            continue                              # формообразующее без форм в БД → пропускаем
        if not fdict:
            continue
        cands.append((e, group, fdict))
    info = {}
    for e, group, fdict in cands:
        if group not in ("noun", "verb", "adjective"):
            continue
        cells = cells_for(e, fdict)
        if cells:
            info[e["row"]["pool_id"]] = {"e": e, "fdict": fdict, "cells": cells}
    return cands, info


def pending_cells(pid, info, fstates):
    """Несданные клетки слова: сперва начатые (они всегда due — рампа держит due=сейчас),
    затем нетронутые (пойдут карточкой). Сданная = produce пройден → interval ≥ 1."""
    started, fresh = [], []
    for c in info[pid]["cells"]:
        st = fstates.get((pid, c))
        if st is None:
            fresh.append(c)
        elif (st.get("interval_days") or 0) < 1:
            started.append(c)
    return started + fresh


def empty_plan(cap_new):
    """План «фазы нет» (грамматика выключена / дрилл по набору): пусто, cap_new НЕТРОНУТ."""
    return {"picks": [], "phase": "words", "cap_new": cap_new, "cards_cap": 0,
            "cycle_left": 0, "cycle_cells": 0, "save_cycle": None}


def plan_forms_phase(cands, info, *, fstates, cycle_state, now_s, cap_new, size,
                     base_servable, batch_size, session_share, overlay_pending):
    """FSM цикла «слова↔формы» + отбор грамм-заданий фазы forms.

    cycle_state — {"phase","batch"} из form_cycle или None (ни разу не создавался);
    base_servable — сколько элементов base-сессия реально отдаст (для анти-дедлока);
    overlay_pending(e, fdict) → несданные клетки местоим-overlay.

    Переходы: seed (None → ветеран с бэклогом ≥ batch_size сразу в forms, новичок в
    words) · партия сдана → words · анти-дедлок words → forms (базе нечего отдать,
    а работа по формам есть). Решение о персисте — в save_cycle (финальное состояние;
    промежуточный seed не персистим отдельно — итог в БД тот же).
    """
    def _freq(p):
        return -(info[p]["e"]["row"].get("freq") or 0)

    def _pending(pid):
        return pending_cells(pid, info, fstates)

    save = None
    cyc = cycle_state
    if cyc is None:
        # Первый заход после релиза цикла: у ветерана уже есть партия несданных форм
        # (бэклог выученных слов) → сразу фаза forms из десятки частотных.
        backlog = sorted((p for p in info if _pending(p)), key=_freq)
        cyc = ({"phase": "forms", "batch": backlog[:batch_size]}
               if len(backlog) >= batch_size else {"phase": "words", "batch": []})
        save = (cyc["phase"], cyc["batch"])
    phase, batch = cyc["phase"], [p for p in cyc["batch"] if p in info]
    batch_pending = [p for p in batch if _pending(p)] if phase == "forms" else []
    if phase == "forms" and not batch_pending:
        phase = "words"                 # партия сдана (или все её клетки выключены
        save = ("words", [])            # тумблерами) → по кругу: снова слова
    # ── ДОСРОЧНЫЙ флип words→forms (анти-дедлок ветерана): базе нечего отдать (повторы
    # не подошли, новые кончились/заблокированы), а работа по формам ЕСТЬ (несданные
    # клетки или подошедшие повторы) → не даём сессии опустеть: переключаемся на формы,
    # добирая партию из бэклога. ──
    if phase == "words":
        has_form_work = any(_pending(p) for p in info) or any(
            pid0 in info and c0 in info[pid0]["cells"]
            and (st0.get("interval_days") or 0) >= 1 and (st0.get("due_at") or "") <= now_s
            for (pid0, c0), st0 in fstates.items())
        if base_servable == 0 and has_form_work:
            extra = sorted((p for p in info if _pending(p) and p not in batch), key=_freq)
            batch = (batch + extra)[:batch_size]
            phase = "forms"
            batch_pending = [p for p in batch if _pending(p)]
            save = ("forms", batch)

    if phase != "forms":
        plan = empty_plan(cap_new)
        plan["save_cycle"] = save
        return plan

    # ФАЗА ФОРМ: новых слов не вводим (cap_new=0 в результате), большинство слотов — формы.
    # КАРТОЧКИ форм — порциями (cards_cap = настройка «новых за сессию»): иначе первая
    # сессия партии — стена из 10-14 карточек подряд без единого задания.
    picks = []   # [(kind 'form'|'overlay', e, cell, fdict, stage|None)]
    cards_cap = max(1, cap_new)
    quota = max(1, round(size * session_share))
    # 0) подошедшие ПОВТОРЫ уже СДАННЫХ клеток (interval≥1, due≤now) — первыми: формы
    #    повторяются в СВОЕЙ фазе (в фазе слов форм нет вовсе)
    due_rev = []
    for (rpid, rc), strow in fstates.items():
        if (rpid in info and rc in info[rpid]["cells"]
                and (strow.get("interval_days") or 0) >= 1 and (strow.get("due_at") or "") <= now_s):
            due_rev.append(((strow.get("due_at") or ""), rpid, rc, strow.get("stage") or "produce"))
    due_rev.sort()
    taken = {}   # pid → взятые клетки этой сессии
    for _d, rpid, rc, rstg in due_rev:
        if len(picks) >= quota:
            break
        picks.append(("form", info[rpid]["e"], rc, info[rpid]["fdict"], rstg))
        taken.setdefault(rpid, set()).add(rc)
    # порядок слов: партия (частотные первыми) → бэклог-филлер (переваривает старые
    # выученные слова, чьи формы ещё не отработаны; флип от филлера не зависит)
    word_order = sorted(batch_pending, key=_freq)
    word_order += sorted((p for p in info if p not in set(batch) and _pending(p)), key=_freq)
    cards_n = 0  # карточек формы уже взято (порция ≤ cards_cap)
    for _round in (0, 1):                      # ≤2 клетки на слово, раунд-робином
        for pid in word_order:
            if len(picks) >= quota:
                break
            got = taken.setdefault(pid, set())
            if len(got) > _round:              # на первом круге по одной, на втором по второй
                continue
            nxt = next((c for c in _pending(pid) if c not in got), None)
            if not nxt:
                continue
            strow = fstates.get((pid, nxt))
            stage = (strow.get("stage") or "card") if strow else "card"
            if stage == "card":
                if cards_n >= cards_cap:       # порция карточек исчерпана → ждём след. сессии
                    continue
                cards_n += 1
            picks.append(("form", info[pid]["e"], nxt, info[pid]["fdict"], stage))
            got.add(nxt)
    # местоим-overlay (курируемая парадигма, вне трека форм) — добивает остаток квоты
    ov_picks = []
    for e, group, fdict in cands:
        if group != "pronoun":
            continue
        pending_ov = overlay_pending(e, fdict)
        if pending_ov:
            ov_picks.append((e["row"].get("freq") or 0, ("overlay", e, pending_ov[0], fdict, None)))
    ov_picks.sort(key=lambda x: -x[0])
    picks += [it for _, it in ov_picks][:max(0, quota - len(picks))]
    return {"picks": picks, "phase": "forms", "cap_new": 0, "cards_cap": cards_cap,
            "cycle_left": len(batch_pending),
            "cycle_cells": sum(len(_pending(pid)) for pid in batch_pending),
            "save_cycle": save}
