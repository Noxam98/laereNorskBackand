"""Регрессии починок раздела «Онлайн» (см. online.py/online_logic.py). Отдельный файл — не трогает
test_online_logic.py. Тесты не требуют БД (save_match падает на отсутствии таблицы → ловится и логается)
и не ходят в сеть (ranked_pool/_candidates замоканы, формы кладём прямо в item)."""
import json
import types
import asyncio

from fastapi import WebSocketDisconnect
import online


class FakeWS:
    """Минимальный дубль WebSocket для драйва ws_online: очередь входящих фреймов, лог исходящих."""
    def __init__(self, incoming, lang="ru"):
        self.query_params = {"token": "x", "lang": lang}
        self._in = list(incoming)
        self.sent = []
        self.closed = None
        self.accepted = False

    async def accept(self):
        self.accepted = True

    async def close(self, code=None):
        self.closed = code

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def receive_text(self):
        if self._in:
            return self._in.pop(0)
        raise WebSocketDisconnect()   # очередь пуста → «клиент отключился»


# ---------------------------------------------------------------- #1: битый фрейм не рвёт сокет
async def test_bad_frames_do_not_drop_socket(monkeypatch):
    async def fake_auth(_token):
        return {"id": 1001, "username": "u", "display_name": ""}
    monkeypatch.setattr(online, "_auth", fake_auth)

    frames = [
        "not json at all",                                      # не-JSON
        "5",                                                    # JSON-число (не объект)
        "[]",                                                   # JSON-массив (msg.get упал бы)
        "null",                                                 # JSON null
        json.dumps({"type": "create", "settings": [1, 2, 3]}),  # settings-список → _norm_settings падает
        json.dumps({"type": "answer", "q": -1, "choice": 0}),   # answer до игры — no-op, не крэш
        json.dumps({"type": "watch"}),                          # ДОЛЖЕН отработать (сокет жив)
    ]
    ws = FakeWS(frames)
    await online.ws_online(ws)

    types_sent = [m.get("type") for m in ws.sent]
    assert types_sent.count("error") >= 4          # битые фреймы → error/bad_message, не разрыв
    assert any(m.get("type") == "rooms" for m in ws.sent)  # watch после битых отработал → сокет жив
    assert ws.closed is None                        # auth ок → сокет не закрывался
    assert online._conns.get(1001, 0) == 0          # слот соединения освобождён в finally


# ---------------------------------------------------------------- #5: кап соединений на юзера
async def test_conn_cap_rejects_excess(monkeypatch):
    async def fake_auth(_token):
        return {"id": 2002, "username": "u", "display_name": ""}
    monkeypatch.setattr(online, "_auth", fake_auth)
    online._conns[2002] = online._MAX_CONNS_PER_USER   # уже на потолке
    try:
        ws = FakeWS([])
        await online.ws_online(ws)
        assert ws.closed == 4409 and not ws.accepted
        assert online._conns[2002] == online._MAX_CONNS_PER_USER  # отказ не тронул счётчик
    finally:
        online._conns.pop(2002, None)


# ---------------------------------------------------------------- #3: keys нет в вопросе
def test_question_payload_has_no_keys():
    q_perlang = {"per_lang": True, "no": "hund",
                 "options": {"ru": ["a", "b", "c", "d"]},
                 "keys": {"ru": ["k1", "k2", "k3", "k4"]}, "correct": {"ru": 0}}
    p = online._q_payload(q_perlang, 0, 3, "ru", 15)
    assert "keys" not in p                              # утечка ответа (keys.index(prompt)) закрыта
    assert p["prompt"] == "hund" and p["options"] == ["a", "b", "c", "d"]

    q_shared = {"per_lang": False, "prompt": {"ru": "собака"},
                "options": ["hund", "katt", "hus", "bil"],
                "keys": ["hund", "katt", "hus", "bil"], "correct": 0}
    p2 = online._q_payload(q_shared, 1, 3, "ru", 15)
    assert "keys" not in p2 and p2["prompt"] == "собака"


# ------------------------------------- #2 + #3: битый choice не морозит комнату, keys в reveal
async def test_bad_choice_does_not_freeze_and_reveal_has_keys(monkeypatch):
    q = {"per_lang": True, "no": "hund",
         "options": {"ru": ["a", "b", "c", "d"]},
         "keys": {"ru": ["k1", "k2", "k3", "k4"]}, "correct": {"ru": 1}}

    async def fake_cand(_room):
        return [1]

    async def fake_build(_cand, _langs, _count, _direction):
        return [q]

    monkeypatch.setattr(online, "_candidates", fake_cand)
    monkeypatch.setattr(online, "_build_quiz", fake_build)
    monkeypatch.setattr(online, "REVEAL_PAUSE", 0)

    user = {"id": 5, "username": "a", "display_name": ""}
    ws = FakeWS([])
    host = online.Player(user, ws, "ru")
    room = online.Room(host, "R", online._norm_settings({"game": "quiz", "source": "pool", "count": 3, "qtime": 5}))
    host.room = room

    task = asyncio.create_task(online.run_quiz(room))
    for _ in range(200):                       # ждём старта первого вопроса
        await asyncio.sleep(0.005)
        if getattr(room, "_cur", -1) == 0 and getattr(room, "_answers", None) is not None:
            break
    room._answers[ws] = ("garbage", 0.5)       # инъекция БИТОГО выбора (раньше ронял votes-loop)
    room._qevent.set()
    await asyncio.wait_for(task, timeout=5)

    sent_types = [m.get("type") for m in ws.sent]
    assert "reveal" in sent_types              # раунд не упал (isinstance-гард в votes)
    assert "ended" in sent_types              # игра дошла до конца → комната не заморожена
    reveal = next(m for m in ws.sent if m.get("type") == "reveal")
    assert reveal.get("keys") == ["k1", "k2", "k3", "k4"]   # #3: keys уехали в reveal
    question = next(m for m in ws.sent if m.get("type") == "question")
    assert "keys" not in question             # #3: в вопросе keys нет
    assert room.state == "lobby"              # #2: finally сбросил комнату в лобби


# ---------------------------------------------------------------- #2: run_quiz finally → лобби
async def test_run_quiz_resets_to_lobby_on_error(monkeypatch):
    async def boom(_room):
        raise RuntimeError("candidates failed")
    monkeypatch.setattr(online, "_candidates", boom)

    user = {"id": 7, "username": "b", "display_name": ""}
    ws = FakeWS([])
    host = online.Player(user, ws, "ru")
    room = online.Room(host, "R", online._norm_settings({"source": "pool"}))
    host.room = room

    await online.run_quiz(room)               # исключение внутри → не пробрасывается
    assert room.state == "lobby"              # finally разморозил
    assert host.ready is False


# ---------------------------------------------------------------- #6: нет дистрактора-синонима
async def test_int2no_skips_synonym_distractor(monkeypatch):
    async def no_neigh(_emb, _no, _k):
        return []
    monkeypatch.setattr(online, "ranked_pool", no_neigh)
    monkeypatch.setattr(online.random, "shuffle", lambda x: None)   # детерминизм: target = cand[0]

    target = {"norwegian": "stor", "translate": {"ru": ["большой"]}, "embedding": None}
    synonym = {"norwegian": "diger", "translate": {"ru": ["большой"]}}   # тот же перевод → 2-й верный
    d1 = {"norwegian": "liten", "translate": {"ru": ["маленький"]}}
    d2 = {"norwegian": "rask", "translate": {"ru": ["быстрый"]}}
    d3 = {"norwegian": "treg", "translate": {"ru": ["медленный"]}}
    cand = [target, synonym, d1, d2, d3]

    qs = await online._build_quiz(cand, ["ru"], 1, "int2no")
    assert len(qs) == 1
    q = qs[0]
    assert q["no"] == "stor"
    assert "diger" not in q["options"]                       # синоним-дистрактор исключён
    assert set(q["options"]) == {"stor", "liten", "rask", "treg"}


# ---------------------------------------------------------------- #8: race int2no принимает форму
def test_race_int2no_accepts_word_form():
    p = types.SimpleNamespace(lang="ru")
    room = types.SimpleNamespace(settings={"answer": "type", "dir": "int2no", "source": "pool"}, host=p)
    item = {"no": "hund", "translate": {"ru": ["собака"]}, "pos": "noun",
            "forms": {"def_sg": "hunden", "indef_pl": "hunder", "def_pl": "hundene"}}

    assert online._race_check(room, item, p, {"text": "hunden"}) is True     # словоформа (опр. ед.)
    assert online._race_check(room, item, p, {"text": "hund"}) is True       # лемма
    assert online._race_check(room, item, p, {"text": "  Hunder "}) is True  # форма + нормализация
    assert online._race_check(room, item, p, {"text": "katt"}) is False      # чужое слово


# ---------------------------------------------------------------- #8: без форм — только лемма (как раньше)
def test_race_int2no_lemma_only_without_forms():
    p = types.SimpleNamespace(lang="ru")
    room = types.SimpleNamespace(settings={"answer": "type", "dir": "int2no", "source": "pool"}, host=p)
    item = {"no": "hund", "translate": {"ru": ["собака"]}, "pos": ""}   # нет forms, нет pos → ordbank пуст
    assert online._race_check(room, item, p, {"text": "hund"}) is True
