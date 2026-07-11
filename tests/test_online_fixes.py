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


# ------------------------------------------------- гонка: падение → подъём (воспроизвести ответ)
def _race_room(answer="type", dir_="int2no"):
    """Комната-гонка с двумя словами и одним игроком на старте (без сокетов и БД)."""
    ws = FakeWS([])
    p = types.SimpleNamespace(ws=ws, lang="ru", user={"id": 1, "username": "u", "display_name": ""},
                              animal="fox", race_queue=[0, 1], race_correct=0, race_state="neutral",
                              race_rank=0, race_token=1, race_fallen=None, streak=0, gone=False)
    words = [{"no": "hund", "translate": {"ru": ["собака"]}, "pos": ""},
             {"no": "katt", "translate": {"ru": ["кошка"]}, "pos": ""}]
    room = types.SimpleNamespace(settings={"answer": answer, "dir": dir_, "source": "pool", "game": "race"},
                                 host=p, players=[p], race_words=words, race_ranks=0,
                                 race_over=asyncio.Event(), race_grace_task=None)
    return room, p, ws


async def test_race_wrong_answer_fells_animal_and_reveals_correct_word():
    """Ошибка → зверь ЛЕЖИТ: сервер шлёт верный ответ и НЕ выдаёт следующее слово."""
    room, p, ws = _race_room()
    await online._race_answer(room, p, {"token": 1, "text": "неверно"})

    res = next(m for m in ws.sent if m["type"] == "race_result")
    assert res["correct"] is False and res["answer"] == "hund"   # показали верное слово
    assert p.race_fallen is not None and p.race_state == "stalled"
    assert not [m for m in ws.sent if m["type"] == "race_word"]   # следующее слово НЕ выдано
    assert [m for m in ws.sent if m["type"] == "race_pos"][-1]["positions"][0]["fallen"] is True


async def test_race_fallen_player_cannot_answer_next_word():
    """Пока лежишь — обычные ответы не принимаются (нельзя «проехать» падение)."""
    room, p, ws = _race_room()
    await online._race_answer(room, p, {"token": 1, "text": "неверно"})
    before = len(ws.sent)
    await online._race_answer(room, p, {"token": 1, "text": "hund"})   # верный, но зверь лежит
    assert len(ws.sent) == before and p.race_correct == 0


async def test_race_recover_requires_correct_word_then_serves_next():
    """Неверный подъём не поднимает; верный — поднимает и только тогда даёт следующее слово."""
    room, p, ws = _race_room()
    await online._race_answer(room, p, {"token": 1, "text": "неверно"})

    await online._race_recover(room, p, {"token": 1, "text": "katt"})   # чужое слово
    bad = [m for m in ws.sent if m["type"] == "race_recover"][-1]
    assert bad["ok"] is False and p.race_fallen is not None
    assert not [m for m in ws.sent if m["type"] == "race_word"]

    await online._race_recover(room, p, {"token": 1, "text": "Hund "})  # верное (нормализуется)
    good = [m for m in ws.sent if m["type"] == "race_recover"][-1]
    assert good["ok"] is True and p.race_fallen is None and p.race_state == "restarting"
    assert [m for m in ws.sent if m["type"] == "race_word"]             # поднялся → поехали дальше


# ------------------------------------------------- AI-набор: перегенерация только по входу подбора
def test_ai_regen_only_on_relevant_settings_change():
    """update_settings перезапускал генерацию на ЛЮБОЕ сохранение настроек AI-комнаты. Теперь —
    только когда изменился вход подбора (источник/уровень/тема/кол-во)."""
    S = online._norm_settings
    base = S({"source": "ai", "level": "A1", "topic": "food", "count": 7, "game": "quiz"})

    # неважные поля (тип игры/лимит/время/приватность/имя) — набор не трогаем
    irrelevant = S({"source": "ai", "level": "A1", "topic": "food", "count": 7,
                    "game": "race", "maxPlayers": 8, "qtime": 30, "private": True})
    assert online._ai_needs_regen(base, irrelevant) is False

    # уровень / тема / количество — перегенерируем
    for chg in ({"level": "B2"}, {"topic": "travel"}, {"count": 12}):
        assert online._ai_needs_regen(base, S({"source": "ai", "level": "A1", "topic": "food",
                                               "count": 7, **chg})) is True

    # включили AI из пула — перегенерируем; сменили AI на пул — нет (набор гасит ветка else)
    assert online._ai_needs_regen(S({"source": "pool"}), base) is True
    assert online._ai_needs_regen(base, S({"source": "pool"})) is False


# ------------------------------------------------- реконнект: слот и состояние держатся в игре
def _player(uid, ws=None, **kw):
    p = types.SimpleNamespace(ws=ws or FakeWS([]), lang="ru", gone=False, reap=None, ready=False,
                              user={"id": uid, "username": f"u{uid}", "display_name": ""},
                              animal="fox", score=0, streak=0, room=None,
                              race_queue=[0, 1], race_correct=0, race_state="neutral",
                              race_rank=0, race_token=1, race_fallen=None)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_transplant_carries_game_state_to_new_socket():
    """Возврат = новый объект Player: всё нажитое (счёт/очередь/упавший зверь) должно переехать."""
    old = _player(1, race_correct=3, race_queue=[2, 0], race_rank=0, score=770, streak=4,
                  race_fallen={"no": "hund"}, race_token=9, animal="lynx")
    new = _player(1)
    online._transplant(old, new)
    assert new.race_correct == 3 and new.race_queue == [2, 0] and new.score == 770
    assert new.streak == 4 and new.race_fallen == {"no": "hund"} and new.race_token == 9
    assert new.animal == "lynx"


def test_live_skips_disconnected_players():
    """Отвалившийся в окне реконнекта не подвешивает раунд/гонку: условия считаются по живым."""
    a, b = _player(1), _player(2, gone=True)
    room = types.SimpleNamespace(players=[a, b])
    assert online._live(room) == [a]


async def test_drop_in_lobby_removes_but_in_game_keeps_slot(monkeypatch):
    """В лобби обрыв = выход. В игре — слот держится (gone), состояние живо, ждём реконнекта."""
    monkeypatch.setattr(online, "_broadcast_rooms", lambda: asyncio.sleep(0))
    monkeypatch.setattr(online, "_send_room", lambda r: asyncio.sleep(0))
    monkeypatch.setattr(online, "RECONNECT_GRACE", 30)

    p = _player(1, race_correct=2)
    lobby = types.SimpleNamespace(id="r1", state="lobby", players=[p], host=p, task=None,
                                  settings={"game": "race", "maxPlayers": 4})
    p.room = lobby
    online._rooms["r1"] = lobby
    await online._drop(p)
    assert p.room is None and p.gone is False and lobby.players == []   # лобби: обычный выход

    q = _player(2, race_correct=5)
    other = _player(3)
    game = types.SimpleNamespace(id="r2", state="playing", players=[q, other], host=q, task=None,
                                 settings={"game": "race", "maxPlayers": 4},
                                 race_over=asyncio.Event(), race_words=[1, 2])
    q.room = game; other.room = game
    online._rooms["r2"] = game
    await online._drop(q)
    try:
        assert q.gone is True and q.room is game           # слот держим
        assert q in game.players and q.race_correct == 5   # состояние живо
        assert online._live(game) == [other]               # но раунд его не ждёт
        assert q.reap is not None
    finally:
        q.reap.cancel()
        online._rooms.pop("r2", None)


async def test_race_positions_show_disconnected_as_dnf():
    """Соперникам отвалившийся показывается «отключился», но прогресс и место сохранены."""
    a = _player(1, race_correct=4, gone=True, race_fallen={"no": "hund"})
    room = types.SimpleNamespace(players=[a], race_words=[1, 2, 3, 4, 5])
    pos = online._race_positions(room)[0]
    assert pos["state"] == "dnf" and pos["progress"] == 4
    assert pos["fallen"] is False      # пока его нет — падение не мигает у соперников


async def test_race_recover_does_not_award_progress():
    """Подъём — обучающий шаг, а не очко: прогресс не растёт, слово вернулось в конец очереди."""
    room, p, _ = _race_room()
    await online._race_answer(room, p, {"token": 1, "text": "неверно"})
    await online._race_recover(room, p, {"token": 1, "text": "hund"})
    assert p.race_correct == 0
    assert p.race_queue == [1, 0]      # промах уехал в конец, следующим идёт katt
