"""Раздел «Онлайн»: многопользовательские комнаты поверх WebSocket (Kahoot-стиль).
Состояние — в памяти (одна машина Fly; при масштабировании нужен внешний брокер).

Поток: создать комнату с настройками → игроки заходят из списка → каждый жмёт «Готов» →
когда все готовы и ≥2 → обратный отсчёт (тиканье на клиенте) → игра. Тип игры —
расширяемый реестр GAMES (сейчас один: quiz). Результаты матчей пишутся в match_log.
"""
import os
import json
import time
import asyncio
import random
import jwt
from collections import defaultdict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from config import logger
from auth import SECRET_KEY, ALGORITHM
from db import get_user, get_pool_duel_words, get_user_quiz_words, save_match, ordbank
from llm import ranked_pool
from autofill import ai_game_words
from fuzzy import word_forms   # поверхностные формы слова (лемма + словоформы) для приёма ответа гонки
from ratelimit import _hit     # in-memory скользящее окно (кап частоты create поверх WS)
# Чистые правила игры (очки, нормализация ответа, кламп настроек, payload вопроса) — без сокетов.
from online_logic import (
    _clamp, _norm_settings, _q_payload, _q_correct, _q_keys, _gain, _norm_answer,
    GAME_KEYS, QUESTION_TIME, MIN_PLAYERS, MAX_PLAYERS_CAP, COUNT_MIN, COUNT_MAX, QTIME_MIN, QTIME_MAX,
)

router = APIRouter()

# --- Анти-абьюз (комнаты/AI жгут LLM-квоту): кап живых WS на юзера + кап частоты create. ---
_MAX_CONNS_PER_USER = int(os.getenv("ONLINE_MAX_CONNS", "3"))   # ≤ N одновременных сокетов на юзера
_CREATE_MAX = int(os.getenv("ONLINE_CREATE_MAX", "10"))         # ≤ N создаваний комнат
_CREATE_WINDOW = int(os.getenv("ONLINE_CREATE_WINDOW", "60"))   # за окно (сек)
_conns = defaultdict(int)   # user_id -> число активных WS (декремент в finally)

# --- Параметры (дефолты; часть берётся из настроек комнаты). Границы клампа — в online_logic. ---
COUNTDOWN = 5           # сек обратного отсчёта перед стартом (тиканье на клиенте)
REVEAL_PAUSE = 4        # сек показа таблицы между вопросами
RACE_GRACE = 25         # сек «добивания» остальным после финиша лидера (гонка)
RACE_MAX = 300          # предохранитель: жёсткий потолок длительности гонки

ANIMALS = ["fox", "hare", "reindeer", "wolf", "elk", "lynx"]  # бегуны гонки (выбор в лобби)

_rooms = {}             # id -> Room
_watchers = set()       # websockets, смотрящие список комнат
_lock = asyncio.Lock()


def _assign_animal(room, player):
    """Первый свободный зверь в комнате (или по индексу, если все заняты)."""
    used = {getattr(p, "animal", None) for p in room.players if p is not player}
    for a in ANIMALS:
        if a not in used:
            return a
    return ANIMALS[len(room.players) % len(ANIMALS)]


def _name(user):
    return (user.get("display_name") or "").strip() or user["username"]


def _plang(room, p):
    """Эффективный язык игрока для сборки вопроса/проверки. AI-набор переводится ТОЛЬКО на язык
    хоста (см. _kick_ai/_candidates), поэтому AI-комната целиком играет на языке набора — иначе
    джойнер другого языка отфильтровывал бы все слова (was: not_enough_words в разноязыких)."""
    if room.settings["source"] == "ai":
        return getattr(room, "ai_lang", None) or room.host.lang
    return p.lang


async def _send(ws, obj):
    try:
        await ws.send_text(json.dumps(obj, ensure_ascii=False))
    except Exception:
        pass


async def _auth(token):
    if not token:
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None
    return await get_user(payload.get("sub"))


def _rid():
    return "".join(random.choices("abcdefghijkmnpqrstuvwxyz23456789", k=6))


class Player:
    def __init__(self, user, ws, lang):
        self.user = user
        self.ws = ws
        self.lang = lang
        self.ready = False
        self.score = 0
        self.streak = 0     # верных ответов подряд (для бонуса и показа «🔥 N»)
        self.animal = None  # выбранный зверь для гонки
        self.room = None


class Room:
    def __init__(self, host, name, settings):
        self.id = _rid()
        self.name = (name or "").strip()[:40] or f"{_name(host.user)}"
        self.settings = settings
        self.players = [host]
        self.host = host
        self.state = "lobby"          # lobby | countdown | playing | ended
        self.task = None              # активная игра/отсчёт
        self.ai_status = None         # для AI-комнаты: generating | indexing | ready | error
        self.ai_words = None          # подготовленный набор слов (AI)
        self.ai_task = None           # задача подготовки AI-набора
        self.ai_lang = None           # язык, на который сгенерён AI-набор (фиксируется при _kick_ai)

    def summary(self):
        return {"id": self.id, "name": self.name, "game": self.settings["game"],
                "answer": self.settings["answer"],
                "players": len(self.players), "max": self.settings["maxPlayers"],
                "level": self.settings["level"] or "", "topic": self.settings["topic"] or "",
                "count": self.settings["count"], "dir": self.settings["dir"], "state": self.state,
                "qtime": self.settings["qtime"], "source": self.settings["source"], "private": self.settings["private"]}

    def detail_for(self, ws):
        return {"type": "room", "room": {
            "id": self.id, "name": self.name, "settings": self.settings, "state": self.state,
            "hostId": self.host.user["id"], "aiStatus": self.ai_status,
            "players": [{"name": _name(p.user), "ready": p.ready, "score": p.score, "animal": p.animal,
                         "isHost": p is self.host, "isYou": p.ws is ws} for p in self.players],
        }}


async def _broadcast_rooms():
    rooms = [r.summary() for r in _rooms.values() if not r.settings["private"]]
    msg = {"type": "rooms", "rooms": rooms}
    for ws in list(_watchers):
        await _send(ws, msg)


async def _send_room(room):
    for p in room.players:
        await _send(p.ws, room.detail_for(p.ws))


async def _maybe_start(room):
    """Если все готовы и игроков ≥ MIN — запустить обратный отсчёт."""
    if room.state != "lobby" or len(room.players) < MIN_PLAYERS:
        return
    if all(p.ready for p in room.players):
        room.state = "countdown"
        await _send_room(room)
        await _broadcast_rooms()
        room.task = asyncio.create_task(_countdown_then_play(room))


async def _countdown_then_play(room):
    try:
        for s in range(COUNTDOWN, 0, -1):
            for p in room.players:
                await _send(p.ws, {"type": "countdown", "sec": s})
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        return
    await GAMES[room.settings["game"]](room)


async def _cancel_countdown(room):
    if room.state == "countdown":
        if room.task:
            room.task.cancel()
        room.state = "lobby"
        await _send_room(room)
        await _broadcast_rooms()


# ----------------------------- Источник слов (общий для quiz/race) -----------------------------

async def _candidates(room):
    """Кандидаты-слова под настройки комнаты: AI-набор (готов в лобби) / словари хоста / общий пул."""
    s = room.settings
    n = max(s["count"] * 8, 60)
    if s["source"] == "ai":      # AI-подбор: набор уже подготовлен в лобби (_prepare_ai)
        ai_lang = getattr(room, "ai_lang", None) or room.host.lang
        return room.ai_words or await ai_game_words(ai_lang, s["level"], s["topic"], s["count"],
                                                     created_by=room.host.user["id"], approved=0)
    if s["source"] == "dict":    # слова из словарей хоста (конкретный по id или все)
        return await get_user_quiz_words(room.host.user["id"], s.get("dictId"), n)
    return await get_pool_duel_words(n, s["level"], s["topic"])  # общий пул по фильтрам


# ----------------------------- Игра: Quiz -----------------------------

async def _distractor_candidates(target, cand):
    """Кандидаты-дистракторы для слова: сначала семантически близкие (по эмбеддингу),
    затем добор случайными из выборки. Формат [{norwegian, translate}]."""
    out = []
    neigh = await ranked_pool(target.get("embedding"), target["norwegian"], 16)
    for w in neigh:  # [{norwegian, data}]
        tr = (w["data"].get("translate") or {}) if isinstance(w.get("data"), dict) else {}
        out.append({"norwegian": w["norwegian"], "translate": tr})
    for w in cand:   # добор/фолбэк (если эмбеддинга нет — это основной источник)
        if w["norwegian"] != target["norwegian"]:
            out.append({"norwegian": w["norwegian"], "translate": w.get("translate", {})})
    return out


async def _build_quiz(cand, langs, count, direction):
    """Список вопросов. Дистракторы — семантически близкие слова (по эмбеддингам), с
    фолбэком на случайные. no2int: варианты-переводы у каждого языка; int2no: варианты —
    норвежские слова (общие), промпт — перевод на языке игрока."""
    qs = []
    pool = list(cand)
    random.shuffle(pool)
    for target in pool:
        if len(qs) >= count:
            break
        if not all((target["translate"].get(l) or []) for l in langs):
            continue
        dcand = await _distractor_candidates(target, pool)
        if direction == "int2no":
            words = []
            for w in dcand:
                if w["norwegian"] == target["norwegian"] or w["norwegian"] in words:
                    continue
                # #6: пропустить дистрактор-синоним — тот, чей перевод совпадает с целью на ЛЮБОМ
                # языке комнаты (иначе он второй ВЕРНЫЙ вариант: промпт = перевод цели → чит/авто-«ошибка»).
                if any((tc := (target["translate"].get(l) or [None])[0]) is not None
                       and (w["translate"].get(l) or [None])[0] == tc for l in langs):
                    continue
                words.append(w["norwegian"])
                if len(words) == 3:
                    break
            if len(words) < 3:
                continue
            options = words + [target["norwegian"]]
            random.shuffle(options)
            qs.append({"per_lang": False, "no": target["norwegian"],
                       "prompt": {l: (target["translate"][l] or [""])[0] for l in langs},
                       "options": options, "keys": options, "correct": options.index(target["norwegian"])})
        else:
            opts, keys, correct, ok = {}, {}, {}, True
            for l in langs:
                ctr = (target["translate"].get(l) or [None])[0]
                if not ctr:
                    ok = False
                    break
                pairs, seen = [], {ctr}   # (norwegian-ключ, перевод-вариант)
                for w in dcand:
                    if w["norwegian"] == target["norwegian"]:
                        continue
                    t = (w["translate"].get(l) or [None])[0]
                    if t and t not in seen:
                        pairs.append((w["norwegian"], t)); seen.add(t)
                    if len(pairs) == 3:
                        break
                if len(pairs) < 3:
                    ok = False
                    break
                pairs.append((target["norwegian"], ctr))
                random.shuffle(pairs)
                opts[l] = [t for _, t in pairs]
                keys[l] = [wd for wd, _ in pairs]   # ключ опции = норвежское слово (общий для всех языков)
                correct[l] = next(idx for idx, (wd, _) in enumerate(pairs) if wd == target["norwegian"])
            if ok:
                qs.append({"per_lang": True, "no": target["norwegian"],
                           "options": opts, "keys": keys, "correct": correct})
    return qs


async def run_quiz(room):
    # #9: инициализируем _answers/_cur/_qstart/_qevent ДО playing — иначе answer c q:-1, пришедший
    # пока готовится AI-набор, падал на отсутствии room._answers. #2: любой сбой раунда обязан
    # разморозить комнату — тело обёрнуто в try/finally с гарантированным _reset_to_lobby.
    room._answers = {}
    room._cur = -1
    room._qstart = time.monotonic()
    room._qevent = asyncio.Event()
    room.state = "playing"
    roster = [{"user_id": p.user["id"], "name": _name(p.user)} for p in room.players]  # #11: кто стартовал
    try:
        for p in room.players:
            p.score = 0
            p.streak = 0
        await _broadcast_rooms()
        s = room.settings
        langs = [_plang(room, room.host)] if s["source"] == "ai" else list({p.lang for p in room.players})
        for p in room.players:   # пока готовим слова (особенно AI-подбор — несколько секунд)
            await _send(p.ws, {"type": "preparing"})
        cand = await _candidates(room)
        questions = await _build_quiz(cand, langs, s["count"], s["dir"])
        if len(questions) < 1:
            for p in room.players:
                await _send(p.ws, {"type": "game_error", "msg": "not_enough_words"})
            return
        qtime = s["qtime"]

        for i, q in enumerate(questions):
            room._answers = {}        # player_ws -> (choice, elapsed)
            room._qevent = asyncio.Event()
            room._qstart = time.monotonic()
            room._cur = i
            for p in room.players:
                await _send(p.ws, _q_payload(q, i, len(questions), _plang(room, p), qtime))
            try:
                await asyncio.wait_for(room._qevent.wait(), timeout=qtime)
            except asyncio.TimeoutError:
                pass
            # скоринг: серия (streak) обновляется по верности, очки — скорость + бонус за серию
            gains = {}
            for p in room.players:
                ans = room._answers.get(p.ws)
                correct = ans is not None and ans[0] == _q_correct(q, _plang(room, p))
                p.streak = p.streak + 1 if correct else 0
                g = _gain(ans[1], correct, p.streak) if correct else 0
                gains[p.ws] = g
                p.score += g
            # голоса по словам-ключам: {норвежское_слово: [имена выбравших]} — клиент разложит по кнопкам
            votes = {}
            for p in room.players:
                ans = room._answers.get(p.ws)
                if ans is not None:
                    kk = _q_keys(q, _plang(room, p))
                    # #2: choice уже приведён к int в хендлере, но гардим — иначе битое значение
                    # роняло бы весь игровой таск (комната зависала в «playing»).
                    if isinstance(ans[0], int) and 0 <= ans[0] < len(kk):
                        votes.setdefault(kk[ans[0]], []).append(_name(p.user))
            standings = _standings(room)
            for p in room.players:
                # #3: keys (норв. ключи опций) шлём тут, на разборе — для маппинга votes→кнопки.
                await _send(p.ws, {"type": "reveal", "correct": _q_correct(q, _plang(room, p)),
                                   "keys": _q_keys(q, _plang(room, p)), "gained": gains[p.ws],
                                   "streak": p.streak, "standings": standings, "votes": votes})
            await asyncio.sleep(REVEAL_PAUSE)

        podium = _standings(room)
        for p in room.players:
            await _send(p.ws, {"type": "ended", "podium": podium})
        # #11: структурный итог в match_log (JSON-блоб — схему не ломаем): ростер + счёт по юзерам + победитель
        results = [{"user_id": p.user["id"], "name": _name(p.user), "score": p.score, "streak": p.streak}
                   for p in sorted(room.players, key=lambda p: -p.score)]
        winner = results[0]["user_id"] if results else None
        try:
            await save_match("quiz", json.dumps({"settings": room.settings, "podium": podium,
                             "roster": roster, "results": results, "winner_id": winner}, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"save_match: {e}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"run_quiz: {e}")
    finally:
        await _reset_to_lobby(room)


def _standings(room):
    ranked = sorted(room.players, key=lambda p: -p.score)
    return [{"name": _name(p.user), "score": p.score, "place": i + 1, "streak": p.streak} for i, p in enumerate(ranked)]


async def _reset_to_lobby(room):
    room.state = "lobby"
    for p in room.players:
        p.ready = False
    await _send_room(room)
    await _broadcast_rooms()


# ----------------------------- Игра: Race (гонка слов) -----------------------------
# Асинхронная (в отличие от quiz): каждый идёт по своему списку в своём темпе, сервер —
# судья (проверяет ответы, чтобы их нельзя было подсмотреть в клиенте) и транслирует
# позиции машин всем. Верно с первого раза → рывок вперёд; ошибка → «глохнет», слово
# уходит в конец очереди. Победитель — первый, кто верно ответил все слова.

async def _build_race(room, cand, langs):
    """Набор для гонки. В режиме «выбор» — те же вопросы, что у quiz (4 варианта).
    В режиме «печать» — список слов [{no, translate}] с переводами на все языки игроков."""
    s = room.settings
    if s["answer"] == "choice":
        return await _build_quiz(cand, langs, s["count"], s["dir"])
    pool = list(cand)
    random.shuffle(pool)
    out = []
    for w in pool:
        if len(out) >= s["count"]:
            break
        if not all((w["translate"].get(l) or []) for l in langs):
            continue   # нужен перевод на всех языках комнаты (и для промпта, и для проверки)
        # pos — для приёма словоформ в int2no (ordbank-lookup, см. _race_accepted_no). У pool-слов
        # он есть; у словарей/AI может отсутствовать (тогда примем только лемму).
        out.append({"no": w["norwegian"], "translate": w["translate"], "pos": w.get("part_of_speech", "")})
    return out


def _race_positions(room):
    # fallen — зверь ЛЕЖИТ и ждёт, пока игрок воспроизведёт верный ответ. Клиент по этому флагу
    # держит анимацию падения (а не гасит её по таймеру), в т.ч. у соперников.
    return [{"id": p.user["id"], "name": _name(p.user), "animal": p.animal,
             "progress": getattr(p, "race_correct", 0), "total": len(room.race_words),
             "state": getattr(p, "race_state", "neutral"),
             "fallen": bool(getattr(p, "race_fallen", None)),
             "finished": bool(getattr(p, "race_rank", 0)),
             "place": getattr(p, "race_rank", 0) or None} for p in room.players]


async def _race_broadcast(room):
    base = _race_positions(room)
    for p in room.players:
        snap = [{**x, "isYou": x["id"] == p.user["id"]} for x in base]
        await _send(p.ws, {"type": "race_pos", "positions": snap})


def _race_standings(room):
    """Финишировавшие — по порядку финиша; недоехавшие — по прогрессу."""
    def key(p):
        return (0, p.race_rank) if getattr(p, "race_rank", 0) else (1, -getattr(p, "race_correct", 0))
    ranked = sorted(room.players, key=key)
    return [{"name": _name(p.user), "place": i + 1, "progress": getattr(p, "race_correct", 0),
             "total": len(room.race_words), "finished": bool(getattr(p, "race_rank", 0)), "animal": p.animal}
            for i, p in enumerate(ranked)]


async def _race_serve(room, p):
    """Отдать игроку текущее слово очереди (с новым токеном — против гонок/двойных ответов)."""
    if not p.race_queue:
        return
    p.race_token += 1
    item = room.race_words[p.race_queue[0]]
    s = room.settings
    lang = _plang(room, p)
    if s["answer"] == "choice":
        # #3: keys (норв. ключи) НЕ шлём в вопросе (в no2int keys.index(prompt) = ответ) — они уедут в race_result.
        pl = _q_payload(item, p.race_correct, len(room.race_words), lang, 0)
        await _send(p.ws, {"type": "race_word", "token": p.race_token, "mode": "choice",
                           "dir": pl["dir"], "prompt": pl["prompt"], "options": pl["options"],
                           "i": p.race_correct, "total": len(room.race_words)})
    else:
        prompt = (item["translate"].get(lang) or [""])[0] if s["dir"] == "int2no" else item["no"]
        await _send(p.ws, {"type": "race_word", "token": p.race_token, "mode": "type",
                           "dir": s["dir"], "prompt": prompt,
                           "i": p.race_correct, "total": len(room.race_words)})


def _race_accepted_no(item):
    """#8: принимаемые формы норвежского ответа в int2no — лемма + словоформы (def_sg/indef_pl/
    present/past/…), а не только голая лемма. Формы берём из item['forms'] (если положили) либо из
    ordbank по (лемма, pos). Нет банка/pos — примем только лемму (как раньше)."""
    forms = item.get("forms")
    if not isinstance(forms, dict):
        forms = ordbank.lookup(item["no"], item.get("pos") or "") or {}
    return word_forms(item["no"], forms)


def _race_check(room, item, p, msg):
    s = room.settings
    lang = _plang(room, p)
    if s["answer"] == "choice":
        try:
            ch = int(msg.get("choice"))
        except (TypeError, ValueError):
            return False   # битый выбор — просто неверно (машина «глохнет»), таск не роняем
        return ch == _q_correct(item, lang)
    if s["dir"] == "int2no":
        accepted = _race_accepted_no(item)      # норв. лемма + словоформы (ordbank)
    else:
        accepted = item["translate"].get(lang) or []
    guess = _norm_answer(msg.get("text"))
    return bool(guess) and guess in {_norm_answer(a) for a in accepted}


async def _race_grace(room):
    """После финиша лидера — окно добивания остальным, затем гонка завершается."""
    leader = next((_name(p.user) for p in room.players if getattr(p, "race_rank", 0) == 1), "")
    try:
        for sec in range(RACE_GRACE, 0, -1):
            for p in room.players:
                await _send(p.ws, {"type": "race_grace", "sec": sec, "leader": leader})
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        return
    room.race_over.set()


def _race_correct_text(room, item, p):
    """Верный ответ СЛОВОМ — показываем упавшему игроку, чтобы он его воспроизвёл и поднял зверя."""
    lang = _plang(room, p)
    if room.settings["answer"] == "choice":
        pl = _q_payload(item, 0, 1, lang, 0)
        return pl["options"][_q_correct(item, lang)]
    if room.settings["dir"] == "int2no":
        return item["no"]
    return (item["translate"].get(lang) or [""])[0]


async def _race_answer(room, p, msg):
    """Обработка ответа игрока в гонке (вызывается из WS-цикла)."""
    if getattr(p, "race_rank", 0) or not getattr(p, "race_queue", None):
        return
    if getattr(p, "race_fallen", None):
        return   # зверь лежит — сначала подъём (race_recover), новые ответы не принимаем
    if msg.get("token") != getattr(p, "race_token", 0):
        return   # устаревший токен (двойной/запоздалый ответ) — игнор
    item = room.race_words[p.race_queue[0]]
    ok = _race_check(room, item, p, msg)
    # #3: keys (норв. ключи опций) шлём в РЕЗУЛЬТАТЕ (не в вопросе) — для разбора на клиенте (choice).
    rkeys = _q_keys(item, _plang(room, p)) if room.settings["answer"] == "choice" else None
    if ok:
        p.race_queue.pop(0)
        p.race_correct += 1
        if not p.race_queue:                     # доехал — фиксируем место
            room.race_ranks += 1
            p.race_rank = room.race_ranks
            p.race_state = "finished"
        else:
            p.race_state = "moving"
        await _send(p.ws, {"type": "race_result", "correct": True, "token": p.race_token, "keys": rkeys})
    else:
        # Ошибка → зверь ПАДАЕТ и лежит, пока игрок не воспроизведёт показанный верный ответ
        # (race_recover). Слово при этом уходит в конец очереди — встретится ещё раз.
        p.race_queue.append(p.race_queue.pop(0))
        p.race_state = "stalled"
        p.race_fallen = item
        await _send(p.ws, {"type": "race_result", "correct": False, "token": p.race_token,
                           "keys": rkeys, "answer": _race_correct_text(room, item, p)})
    await _race_broadcast(room)
    # следующее слово — только если стоим на ногах (упавшему его выдаст _race_recover)
    if not getattr(p, "race_fallen", None) and p.race_state != "finished":
        await _race_serve(room, p)
    # условия завершения гонки
    if all(getattr(q, "race_rank", 0) for q in room.players):
        room.race_over.set()
    elif room.race_ranks >= 1 and getattr(room, "race_grace_task", None) is None:
        room.race_grace_task = asyncio.create_task(_race_grace(room))


async def _race_recover(room, p, msg):
    """Подъём упавшего зверя: игрок обязан воспроизвести показанный верный ответ (ввести/выбрать).
    Проверяет СЕРВЕР — клиент лишь показывает слово, поэтому «поднять» обманом нельзя.
    Не совпало — лежим дальше (можно пробовать сколько угодно, это обучающий шаг, не наказание)."""
    item = getattr(p, "race_fallen", None)
    if not item or getattr(p, "race_rank", 0):
        return
    if msg.get("token") != getattr(p, "race_token", 0):
        return   # токен упавшего слова (новый ещё не выдан) — защита от запоздалых кадров
    if not _race_check(room, item, p, msg):
        await _send(p.ws, {"type": "race_recover", "ok": False, "token": p.race_token})
        return
    p.race_fallen = None
    p.race_state = "restarting"                  # встал → клиент играет подъём
    await _send(p.ws, {"type": "race_recover", "ok": True, "token": p.race_token})
    await _race_broadcast(room)
    await _race_serve(room, p)                   # и только теперь — следующее слово


async def run_race(room):
    # #2/#4: тело в try/finally — любой сбой раунда обязан разморозить комнату И отменить grace-таск
    # (иначе CancelledError на race_over.wait минует cancel и таск сливает Gemini-квоту вхолостую).
    room.race_grace_task = None
    room.race_over = asyncio.Event()
    room.state = "playing"
    roster = [{"user_id": p.user["id"], "name": _name(p.user)} for p in room.players]  # #11: кто стартовал
    try:
        await _broadcast_rooms()
        s = room.settings
        langs = [_plang(room, room.host)] if s["source"] == "ai" else list({p.lang for p in room.players})
        for p in room.players:
            await _send(p.ws, {"type": "preparing"})
        cand = await _candidates(room)
        words = await _build_race(room, cand, langs)
        if len(words) < 2:
            for p in room.players:
                await _send(p.ws, {"type": "game_error", "msg": "not_enough_words"})
            return
        room.race_words = words
        room.race_ranks = 0
        queue0 = list(range(len(words)))   # один и тот же порядок для всех — честная гонка
        for p in room.players:
            p.race_queue = list(queue0)
            p.race_correct = 0
            p.race_state = "neutral"
            p.race_rank = 0
            p.race_token = 0
            p.race_fallen = None    # зверь на ногах (иначе рестарт гонки унаследовал бы падение)
        for p in room.players:
            await _send(p.ws, {"type": "race_go", "total": len(words)})
        await _race_broadcast(room)
        for p in room.players:
            await _race_serve(room, p)
        try:
            await asyncio.wait_for(room.race_over.wait(), timeout=RACE_MAX)
        except asyncio.TimeoutError:
            pass
        podium = _race_standings(room)
        for p in room.players:
            await _send(p.ws, {"type": "ended", "game": "race", "podium": podium})
        # #11: структурный итог в match_log (JSON-блоб): ростер + прогресс/место по юзерам + победитель
        def _rk(p):
            return (0, p.race_rank) if getattr(p, "race_rank", 0) else (1, -getattr(p, "race_correct", 0))
        results = [{"user_id": p.user["id"], "name": _name(p.user), "progress": getattr(p, "race_correct", 0),
                    "rank": getattr(p, "race_rank", 0) or None} for p in sorted(room.players, key=_rk)]
        winner = results[0]["user_id"] if results else None
        try:
            await save_match("race", json.dumps({"settings": room.settings, "podium": podium,
                             "roster": roster, "results": results, "winner_id": winner}, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"save_match: {e}")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        logger.warning(f"run_race: {e}")
    finally:
        gt = getattr(room, "race_grace_task", None)
        if gt:
            gt.cancel()   # #4: гарантированная отмена окна добивания (в т.ч. при cancel/exception)
        await _reset_to_lobby(room)


# Реестр игр — добавлять новые типы сюда (ключ → корутина run(room)).
GAMES = {"quiz": run_quiz, "race": run_race}   # тип игры → async-runner (диспетчер сокет-потока)
assert set(GAMES) == set(GAME_KEYS), "GAMES (runner'ы) рассинхронились с GAME_KEYS (валидатор настроек)"


# ----------------------------- AI-набор слов (готовится в лобби) -----------------------------

async def _prepare_ai(room):
    """Сгенерировать AI-набор слов в лобби с прогрессом: generating → indexing → ready.
    «Готов»/старт заблокированы, пока статус не ready."""
    s = room.settings

    async def phase(p):
        room.ai_status = p
        await _send_room(room)

    try:
        room.ai_status = "generating"
        await _send_room(room)
        words = await ai_game_words(room.host.lang, s["level"], s["topic"], s["count"], on_phase=phase,
                                    created_by=room.host.user["id"], approved=0)
        if asyncio.current_task().cancelled():
            return
        room.ai_words = words
        room.ai_status = "ready" if words else "error"
    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.warning(f"_prepare_ai: {e}")
        room.ai_status = "error"
        room.ai_words = None
    await _send_room(room)


def _kick_ai(room):
    """(Пере)запустить подготовку AI-набора. Сбрасывает готовность игроков (набор сменился)."""
    if room.ai_task and not room.ai_task.done():
        room.ai_task.cancel()
    room.ai_words = None
    room.ai_status = "generating"
    room.ai_lang = room.host.lang   # #7: фиксируем язык набора — на нём и играет вся AI-комната
    for p in room.players:
        p.ready = False
    room.ai_task = asyncio.create_task(_prepare_ai(room))


# ----------------------------- Подключения -----------------------------

async def _leave(me):
    room = me.room
    if not room:
        return
    me.room = None
    if me in room.players:
        room.players.remove(me)
    if not room.players:
        # #4: гасим ВСЕ фоновые таски комнаты (не только game task) — иначе AI-генерация/окно
        # добивания продолжают жечь Gemini-квоту после исчезновения комнаты.
        for tsk in (room.task, getattr(room, "ai_task", None), getattr(room, "race_grace_task", None)):
            if tsk:
                tsk.cancel()
        _rooms.pop(room.id, None)
        await _broadcast_rooms()
        return
    if room.host is me:
        room.host = room.players[0]
    if room.state == "countdown":
        await _cancel_countdown(room)
    # гонка: ушедший игрок не должен подвешивать комнату — пересчитываем условие финиша
    if room.state == "playing" and room.settings["game"] == "race" and getattr(room, "race_over", None):
        await _race_broadcast(room)
        if all(getattr(q, "race_rank", 0) for q in room.players):
            room.race_over.set()
    # #10: quiz — если после ухода игрока ответили ВСЕ оставшиеся, завершаем вопрос досрочно
    # (условие all-answered раньше проверялось только в answer-хендлере → раунд ждал таймаут зря).
    if room.state == "playing" and room.settings["game"] == "quiz" and getattr(room, "_qevent", None):
        if room.players and all(p.ws in room._answers for p in room.players):
            room._qevent.set()
    await _send_room(room)
    await _broadcast_rooms()


@router.websocket("/ws/online")
async def ws_online(ws: WebSocket):
    user = await _auth(ws.query_params.get("token"))
    if not user:
        await ws.close(code=4401)
        return
    uid = user["id"]
    # #5: кап одновременных сокетов на юзера (спам-вкладки/скрипты жгут комнаты и LLM-квоту).
    if _conns[uid] >= _MAX_CONNS_PER_USER:
        await ws.close(code=4409)   # too many connections
        return
    await ws.accept()
    _conns[uid] += 1
    me = Player(user, ws, ws.query_params.get("lang") or "ru")
    try:
        while True:
            # #1: получение фрейма и его обработка — РАЗДЕЛЬНО. Разрыв (WebSocketDisconnect на
            # receive) выходит из while; любой битый фрейм (не-JSON / не-объект / кривой payload)
            # НЕ рвёт сокет и не выкидывает из комнаты — шлём {"type":"error","msg":"bad_message"}.
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await _send(ws, {"type": "error", "msg": "bad_message"})
                continue
            if not isinstance(msg, dict):
                await _send(ws, {"type": "error", "msg": "bad_message"})
                continue
            t = msg.get("type")
            try:
                if t == "watch":
                    _watchers.add(ws)
                    await _send(ws, {"type": "rooms", "rooms": [r.summary() for r in _rooms.values() if not r.settings["private"]]})

                elif t == "unwatch":
                    _watchers.discard(ws)

                elif t == "create":
                    try:   # #5: кап частоты create (спам-создание/уничтожение комнат жжёт AI-квоту)
                        _hit(("online_create", uid), _CREATE_MAX, _CREATE_WINDOW)
                    except HTTPException:
                        await _send(ws, {"type": "error", "msg": "rate_limited"})
                        continue
                    async with _lock:
                        if me.room:
                            await _leave(me)
                        room = Room(me, msg.get("name"), _norm_settings(msg.get("settings")))
                        me.room = room
                        me.animal = _assign_animal(room, me)
                        _rooms[room.id] = room
                        if room.settings["source"] == "ai":
                            _kick_ai(room)   # начать подготовку набора сразу в лобби
                    await _send_room(room)
                    await _broadcast_rooms()

                elif t == "join":
                    async with _lock:
                        room = _rooms.get(msg.get("roomId"))
                        if not room:
                            await _send(ws, {"type": "error", "msg": "room_not_found"})
                        elif room.state != "lobby":
                            await _send(ws, {"type": "error", "msg": "room_in_game"})
                        elif len(room.players) >= room.settings["maxPlayers"]:
                            await _send(ws, {"type": "error", "msg": "room_full"})
                        elif me.room is room:
                            pass
                        else:
                            if me.room:
                                await _leave(me)
                            me.ready = False
                            me.room = room
                            room.players.append(me)
                            me.animal = _assign_animal(room, me)
                    if me.room:
                        await _send_room(me.room)
                        await _broadcast_rooms()

                elif t == "update_settings":
                    room = me.room
                    # менять настройки может только хост и только в лобби; хост при выходе
                    # владельца переназначается оставшемуся (см. _leave).
                    if room and room.host is me and room.state == "lobby":
                        room.settings = _norm_settings(msg.get("settings"))
                        nm = (msg.get("name") or "").strip()[:40]
                        if nm:
                            room.name = nm
                        if room.settings["source"] == "ai":
                            _kick_ai(room)   # настройки сменились → перегенерировать набор
                        else:
                            if room.ai_task and not room.ai_task.done():
                                room.ai_task.cancel()
                            room.ai_status = None
                            room.ai_words = None
                        await _send_room(room)
                        await _broadcast_rooms()

                elif t == "pick_animal":
                    # выбор зверя в лобби (гонка); дубликаты разрешены, по умолчанию все разные
                    if me.room and me.room.state == "lobby" and msg.get("animal") in ANIMALS:
                        me.animal = msg.get("animal")
                        await _send_room(me.room)

                elif t == "ready":
                    # для AI-комнаты нельзя готовиться, пока набор не готов
                    if me.room and me.room.state == "lobby" and not (me.room.settings["source"] == "ai" and me.room.ai_status != "ready"):
                        me.ready = bool(msg.get("ready"))
                        await _send_room(me.room)
                        await _maybe_start(me.room)

                elif t == "retry_ai":
                    # повторить генерацию AI-набора после ошибки (хост, в лобби)
                    room = me.room
                    if room and room.host is me and room.state == "lobby" and room.settings["source"] == "ai":
                        _kick_ai(room)
                        await _send_room(room)
                        await _broadcast_rooms()

                elif t == "force_start":
                    # хост может стартовать вручную (для теста) — без требования «все готовы»/≥2
                    room = me.room
                    if room and room.host is me and room.state == "lobby" and room.players \
                            and not (room.settings["source"] == "ai" and room.ai_status != "ready"):
                        room.state = "countdown"
                        await _send_room(room)
                        await _broadcast_rooms()
                        room.task = asyncio.create_task(_countdown_then_play(room))

                elif t == "leave":
                    async with _lock:
                        await _leave(me)
                    await _send(ws, {"type": "left"})

                elif t == "set_lang":
                    # смена языка интерфейса на живом сокете (фронт не пересоздаёт соединение) —
                    # влияет на язык вопросов игрока (_plang) при следующей сборке.
                    lang = (msg.get("lang") or "").strip()[:8]
                    if lang:
                        me.lang = lang

                elif t == "rejoin":
                    # реконнект: фронт после обрыва просит вернуть в активную комнату по id.
                    # Только лобби (в идущую игру чисто вернуть нельзя — ресинк вне охвата) → иначе
                    # отправляем в браузер комнат. Дедуп по user_id: убираем застрявшую запись с
                    # мёртвым сокетом (если старый _leave ещё не отработал), чтобы не задвоить игрока.
                    room = _rooms.get(msg.get("roomId"))
                    if room and room.state == "lobby" and me.room is not room:
                        async with _lock:
                            if me.room:
                                await _leave(me)
                            room.players = [p for p in room.players if p.user["id"] != uid]
                            if len(room.players) < room.settings["maxPlayers"]:
                                me.ready = False
                                me.room = room
                                room.players.append(me)
                                me.animal = _assign_animal(room, me)
                        if me.room is room:
                            await _send_room(room)
                            await _broadcast_rooms()
                        else:
                            await _send(ws, {"type": "left"})
                    else:
                        await _send(ws, {"type": "left"})

                elif t == "answer" and me.room and me.room.settings["game"] == "race":
                    room = me.room
                    if room.state == "playing":
                        await _race_answer(room, me, msg)

                elif t == "race_recover" and me.room and me.room.settings["game"] == "race":
                    room = me.room       # подъём упавшего зверя: воспроизвести верный ответ
                    if room.state == "playing":
                        await _race_recover(room, me, msg)

                elif t == "answer":
                    room = me.room
                    if room and room.state == "playing" and msg.get("q") == getattr(room, "_cur", -1):
                        # #2: choice приводим к int; битое значение (None/строка/список) — отбрасываем,
                        # иначе оно доходит до скоринга/votes и роняет игровой таск (комната в «playing»).
                        try:
                            choice = int(msg.get("choice"))
                        except (TypeError, ValueError):
                            choice = None
                        if choice is not None and me.ws not in room._answers:
                            room._answers[me.ws] = (choice, time.monotonic() - room._qstart)
                            # сообщить всем, КТО уже ответил (без выбора) — для живой подсветки игроков
                            names = [_name(pl.user) for pl in room.players if pl.ws in room._answers]
                            for pl in room.players:
                                await _send(pl.ws, {"type": "answered", "names": names})
                            if all(p.ws in room._answers for p in room.players):
                                room._qevent.set()
            except WebSocketDisconnect:
                raise   # разрыв — наверх, выходим из while (не глушим как «битый фрейм»)
            except Exception as e:
                # #1: сбой обработки ОДНОГО сообщения не рвёт сокет и не выкидывает из комнаты
                logger.warning(f"ws_online msg [{t}]: {e}")
                await _send(ws, {"type": "error", "msg": "bad_message"})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"ws_online: {e}")
    finally:
        _watchers.discard(ws)
        _conns[uid] -= 1                 # #5: освобождаем слот соединения
        if _conns[uid] <= 0:
            _conns.pop(uid, None)
        async with _lock:
            await _leave(me)
