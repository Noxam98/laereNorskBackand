"""Раздел «Онлайн»: многопользовательские комнаты поверх WebSocket (Kahoot-стиль).
Состояние — в памяти (одна машина Fly; при масштабировании нужен внешний брокер).

Поток: создать комнату с настройками → игроки заходят из списка → каждый жмёт «Готов» →
когда все готовы и ≥2 → обратный отсчёт (тиканье на клиенте) → игра. Тип игры —
расширяемый реестр GAMES (сейчас один: quiz). Результаты матчей пишутся в match_log.
"""
import json
import time
import math
import asyncio
import random
import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from config import logger
from auth import SECRET_KEY, ALGORITHM
from db import get_user, get_pool_duel_words, get_user_quiz_words, save_match
from llm import ranked_pool

router = APIRouter()

# --- Параметры (дефолты; часть берётся из настроек комнаты) ---
QUESTION_TIME = 15      # дефолт сек на вопрос (если не задано в настройках комнаты)
COUNTDOWN = 5           # сек обратного отсчёта перед стартом (тиканье на клиенте)
REVEAL_PAUSE = 4        # сек показа таблицы между вопросами
MIN_PLAYERS = 2
MAX_PLAYERS_CAP = 8
COUNT_MIN, COUNT_MAX = 3, 20
QTIME_MIN, QTIME_MAX = 5, 30   # границы длительности вопроса

_rooms = {}             # id -> Room
_watchers = set()       # websockets, смотрящие список комнат
_lock = asyncio.Lock()


def _name(user):
    return (user.get("display_name") or "").strip() or user["username"]


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


def _clamp(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _norm_settings(s):
    s = s or {}
    return {
        "game": s.get("game") if s.get("game") in GAMES else "quiz",
        "dir": "int2no" if s.get("dir") == "int2no" else "no2int",
        "source": "dict" if s.get("source") == "dict" else "pool",   # pool=общий пул, dict=словари хоста
        "dictId": int(s["dictId"]) if str(s.get("dictId") or "").isdigit() else None,  # None=все словари хоста
        "level": (s.get("level") or "") or None,     # A1..C2 или None=любой
        "topic": (s.get("topic") or "") or None,     # ключ темы или None=любая
        "count": _clamp(s.get("count"), COUNT_MIN, COUNT_MAX, 7),
        "qtime": _clamp(s.get("qtime"), QTIME_MIN, QTIME_MAX, QUESTION_TIME),
        "maxPlayers": _clamp(s.get("maxPlayers"), MIN_PLAYERS, MAX_PLAYERS_CAP, 4),
        "private": bool(s.get("private")),
    }


class Player:
    def __init__(self, user, ws, lang):
        self.user = user
        self.ws = ws
        self.lang = lang
        self.ready = False
        self.score = 0
        self.streak = 0     # верных ответов подряд (для бонуса и показа «🔥 N»)
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

    def summary(self):
        return {"id": self.id, "name": self.name, "game": self.settings["game"],
                "players": len(self.players), "max": self.settings["maxPlayers"],
                "level": self.settings["level"] or "", "topic": self.settings["topic"] or "",
                "count": self.settings["count"], "dir": self.settings["dir"], "state": self.state,
                "qtime": self.settings["qtime"], "source": self.settings["source"], "private": self.settings["private"]}

    def detail_for(self, ws):
        return {"type": "room", "room": {
            "id": self.id, "name": self.name, "settings": self.settings, "state": self.state,
            "hostId": self.host.user["id"],
            "players": [{"name": _name(p.user), "ready": p.ready, "score": p.score,
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
                if w["norwegian"] != target["norwegian"] and w["norwegian"] not in words:
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


async def run_quiz(room):
    room.state = "playing"
    for p in room.players:
        p.score = 0
        p.streak = 0
    await _broadcast_rooms()
    s = room.settings
    langs = list({p.lang for p in room.players})
    n = max(s["count"] * 8, 60)
    if s["source"] == "dict":   # слова из словарей хоста (конкретный по id или все)
        cand = await get_user_quiz_words(room.host.user["id"], s.get("dictId"), n)
    else:                        # общий пул по фильтрам
        cand = await get_pool_duel_words(n, s["level"], s["topic"])
    questions = await _build_quiz(cand, langs, s["count"], s["dir"])
    if len(questions) < 1:
        for p in room.players:
            await _send(p.ws, {"type": "game_error", "msg": "not_enough_words"})
        await _reset_to_lobby(room)
        return
    qtime = s["qtime"]

    for i, q in enumerate(questions):
        room._answers = {}        # player_ws -> (choice, elapsed)
        room._qevent = asyncio.Event()
        room._qstart = time.monotonic()
        room._cur = i
        for p in room.players:
            await _send(p.ws, _q_payload(q, i, len(questions), p.lang, qtime))
        try:
            await asyncio.wait_for(room._qevent.wait(), timeout=qtime)
        except asyncio.TimeoutError:
            pass
        # скоринг: серия (streak) обновляется по верности, очки — скорость + бонус за серию
        gains = {}
        for p in room.players:
            ans = room._answers.get(p.ws)
            correct = ans is not None and ans[0] == _q_correct(q, p.lang)
            p.streak = p.streak + 1 if correct else 0
            g = _gain(ans[1], correct, p.streak) if correct else 0
            gains[p.ws] = g
            p.score += g
        # голоса по словам-ключам: {норвежское_слово: [имена выбравших]} — клиент разложит по кнопкам
        votes = {}
        for p in room.players:
            ans = room._answers.get(p.ws)
            if ans is not None:
                kk = _q_keys(q, p.lang)
                if 0 <= ans[0] < len(kk):
                    votes.setdefault(kk[ans[0]], []).append(_name(p.user))
        standings = _standings(room)
        for p in room.players:
            await _send(p.ws, {"type": "reveal", "correct": _q_correct(q, p.lang),
                               "gained": gains[p.ws], "streak": p.streak, "standings": standings, "votes": votes})
        await asyncio.sleep(REVEAL_PAUSE)

    podium = _standings(room)
    for p in room.players:
        await _send(p.ws, {"type": "ended", "podium": podium})
    try:
        await save_match("quiz", json.dumps({"settings": room.settings, "podium": podium}, ensure_ascii=False))
    except Exception as e:
        logger.warning(f"save_match: {e}")
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


# Реестр игр — добавлять новые типы сюда (ключ → корутина run(room)).
GAMES = {"quiz": run_quiz}


# ----------------------------- Подключения -----------------------------

async def _leave(me):
    room = me.room
    if not room:
        return
    me.room = None
    if me in room.players:
        room.players.remove(me)
    if not room.players:
        if room.task:
            room.task.cancel()
        _rooms.pop(room.id, None)
        await _broadcast_rooms()
        return
    if room.host is me:
        room.host = room.players[0]
    if room.state == "countdown":
        await _cancel_countdown(room)
    await _send_room(room)
    await _broadcast_rooms()


@router.websocket("/ws/online")
async def ws_online(ws: WebSocket):
    user = await _auth(ws.query_params.get("token"))
    if not user:
        await ws.close(code=4401)
        return
    await ws.accept()
    me = Player(user, ws, ws.query_params.get("lang") or "ru")
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            t = msg.get("type")

            if t == "watch":
                _watchers.add(ws)
                await _send(ws, {"type": "rooms", "rooms": [r.summary() for r in _rooms.values() if not r.settings["private"]]})

            elif t == "unwatch":
                _watchers.discard(ws)

            elif t == "create":
                async with _lock:
                    if me.room:
                        await _leave(me)
                    room = Room(me, msg.get("name"), _norm_settings(msg.get("settings")))
                    me.room = room
                    _rooms[room.id] = room
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
                    await _send_room(room)
                    await _broadcast_rooms()

            elif t == "ready":
                if me.room and me.room.state == "lobby":
                    me.ready = bool(msg.get("ready"))
                    await _send_room(me.room)
                    await _maybe_start(me.room)

            elif t == "leave":
                async with _lock:
                    await _leave(me)
                await _send(ws, {"type": "left"})

            elif t == "answer":
                room = me.room
                if room and room.state == "playing" and msg.get("q") == getattr(room, "_cur", -1):
                    if me.ws not in room._answers:
                        room._answers[me.ws] = (msg.get("choice"), time.monotonic() - room._qstart)
                        # сообщить всем, КТО уже ответил (без выбора) — для живой подсветки игроков
                        names = [_name(pl.user) for pl in room.players if pl.ws in room._answers]
                        for pl in room.players:
                            await _send(pl.ws, {"type": "answered", "names": names})
                        if all(p.ws in room._answers for p in room.players):
                            room._qevent.set()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"ws_online: {e}")
    finally:
        _watchers.discard(ws)
        async with _lock:
            await _leave(me)
