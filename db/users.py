import json
import aiosqlite
from .core import _conn, _release, _now


async def get_user(username: str):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await _release(db)


async def username_taken_ci(username: str) -> bool:
    """Регистронезависимая проверка занятости имени. users.username UNIQUE — BINARY (регистрозависим),
    а is_admin сравнивает имя по .lower(): без этой проверки «maksym» и «Maksym» — РАЗНЫЕ аккаунты, и
    оба матчат админ-список ⇒ эскалация привилегий (регистрируешь регистро-вариант админа → ты админ).
    Запрещаем создавать любой регистро-вариант уже занятого имени, чтобы имя было CI-уникальным."""
    db = await _conn()
    try:
        async with db.execute(
                "SELECT 1 FROM users WHERE username = ? COLLATE NOCASE LIMIT 1", (username.strip(),)) as cur:
            return (await cur.fetchone()) is not None
    finally:
        await _release(db)


async def set_user_theme(user_id: int, theme: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET theme = ? WHERE id = ?", (theme, user_id))
        await db.commit()
    finally:
        await _release(db)


async def set_user_focus_topics(user_id: int, topics):
    """topics — список ключей тем (валидируется снаружи). Хранится JSON-массивом; [] = без фокуса."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET focus_topics = ? WHERE id = ?",
                         (json.dumps(topics or [], ensure_ascii=False), user_id))
        await db.commit()
    finally:
        await _release(db)


async def get_user_focus_topics(user_id: int):
    db = await _conn()
    try:
        async with db.execute("SELECT focus_topics FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    if not row or not row["focus_topics"]:
        return []
    try:
        v = json.loads(row["focus_topics"])
        return [t for t in v if isinstance(t, str)] if isinstance(v, list) else []
    except Exception:
        return []


async def get_user_new_per_session(user_id: int, default: int = 6):
    """Сколько НОВЫХ карточек-знакомств вводить за сессию (gamePrefs.newPerSession). Клампим 1..20.
    Пусто/битое → default. Управляет порционностью знакомства со словами (настройка профиля)."""
    db = await _conn()
    try:
        async with db.execute("SELECT game_prefs FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    if not row or not row["game_prefs"]:
        return default
    try:
        v = json.loads(row["game_prefs"]).get("newPerSession")
        if isinstance(v, (int, float)):
            return max(1, min(20, int(v)))
    except Exception:
        pass
    return default


async def get_user_grammar(user_id: int, default: bool = True):
    """Включён ли грамм-overlay (gamePrefs.grammar) — род/формы поверх выученных слов. Дефолт — вкл.
    Пусто/битое → default. Тумблер профиля; не влияет на base-рампу и «выучено»/CEFR."""
    db = await _conn()
    try:
        async with db.execute("SELECT game_prefs FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    if not row or not row["game_prefs"]:
        return default
    try:
        v = json.loads(row["game_prefs"]).get("grammar")
        if isinstance(v, bool):
            return v
    except Exception:
        pass
    return default


_GRAMMAR_POS_KEYS = ("noun", "verb", "adjective", "pronoun")


async def get_user_grammar_pos(user_id: int):
    """Пер-POS тумблеры грамм-overlay (gamePrefs.grammarPos): какие части речи дриллить. Отсутствует/
    битое → все включены. Группы: noun/verb/adjective/pronoun (pronoun = местоимения + притяжательные)."""
    db = await _conn()
    try:
        async with db.execute("SELECT game_prefs FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    out = {k: True for k in _GRAMMAR_POS_KEYS}
    if not row or not row["game_prefs"]:
        return out
    try:
        gp = json.loads(row["game_prefs"]).get("grammarPos")
        if isinstance(gp, dict):
            for k in _GRAMMAR_POS_KEYS:
                if isinstance(gp.get(k), bool):
                    out[k] = gp[k]
    except Exception:
        pass
    return out


def _prefs_of(row):
    """Разбор колонки game_prefs строки users (пусто/битое → {}) — один дом парсинга тумблеров."""
    raw = row["game_prefs"] if row else None
    try:
        gp = json.loads(raw) if raw else {}
    except Exception:
        return {}
    return gp if isinstance(gp, dict) else {}


def _audio_of(gp):
    """Аудиозадания из gamePrefs: audio (дефолт ВКЛ) + бэк-совместимость старого listenOff=true."""
    v = gp.get("audio")
    if isinstance(v, bool):
        return v
    return gp.get("listenOff") is not True


def _choice_of(gp):
    """Ступень «выбор из вариантов» из gamePrefs (choiceStage, дефолт ВКЛ)."""
    v = gp.get("choiceStage")
    return v if isinstance(v, bool) else True


async def _game_prefs_row(user_id: int):
    db = await _conn()
    try:
        async with db.execute("SELECT game_prefs FROM users WHERE id = ?", (user_id,)) as cur:
            return _prefs_of(await cur.fetchone())
    finally:
        await _release(db)


async def get_user_audio(user_id: int):
    """Аудиозадания (слуховые сессии): (включены?, порог партии 5..20). gamePrefs.audio (дефолт — вкл;
    бэк-совместимость: старый listenOff=true → выкл) + gamePrefs.listenPack (клампим 5..20, дефолт 10).
    audio ВКЛ → choice_no2int откладывается в слуховую сессию; ВЫКЛ → идёт в дневной рампе текстом."""
    gp = await _game_prefs_row(user_id)
    pack = 10
    p = gp.get("listenPack")
    if isinstance(p, (int, float)):
        pack = max(5, min(20, int(p)))
    return _audio_of(gp), pack


async def get_user_choice(user_id: int):
    """Включена ли ступень «выбор из вариантов» в рампе (gamePrefs.choiceStage). Дефолт — вкл.
    ВЫКЛ → next_step не выдаёт клетки выбора (слово идёт сразу на продукцию: сборка/ввод, у фраз —
    порядок слов), а сдача более сложной ступени засчитывает их (srs.cells.skipped_choice).
    «Выучено»/CEFR/пачка экзамена по-прежнему считаются по ПОЛНОЙ рампе — тумблер их не двигает."""
    return _choice_of(await _game_prefs_row(user_id))


async def get_user_ramp(user_id: int):
    """Оба тумблера рампы ОДНИМ чтением: (аудиозадания, ступень выбора). Для горячего пути
    apply_result — там нужны оба, а ходить в users дважды за один ответ незачем."""
    gp = await _game_prefs_row(user_id)
    return _audio_of(gp), _choice_of(gp)


async def set_user_game_prefs(user_id: int, prefs_json: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_prefs = ? WHERE id = ?", (prefs_json, user_id))
        await db.commit()
    finally:
        await _release(db)


async def set_user_current_dict(user_id: int, name: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET current_dict = ? WHERE id = ?", (name, user_id))
        await db.commit()
    finally:
        await _release(db)


async def create_user(username: str, hashed_password: str):
    db = await _conn()
    try:
        cur = await db.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        user_id = cur.lastrowid
        # стартовый словарь
        await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, "default", _now()))
        await db.commit()
        return {"message": "User created successfully", "user_id": user_id}
    except aiosqlite.IntegrityError:
        return {"error": "Username already exists"}
    finally:
        await _release(db)


async def set_user_password(user_id: int, hashed_password: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_password, user_id))
        await db.commit()
    finally:
        await _release(db)


async def get_user_by_google_sub(google_sub: str):
    db = await _conn()
    try:
        async with db.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None
    finally:
        await _release(db)


async def set_user_game_mode(user_id: int, mode: str):
    """Запомнить последний выбранный режим в хабе «Игры» (solo|online)."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET game_mode = ? WHERE id = ?", (mode, user_id))
        await db.commit()
    finally:
        await _release(db)


async def set_online_prefs(user_id: int, prefs_json: str):
    """Запомнить последние настройки онлайн-комнаты (чтобы не настраивать каждый раз)."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET online_prefs = ? WHERE id = ?", (prefs_json, user_id))
        await db.commit()
    finally:
        await _release(db)


async def save_match(game: str, data_json: str):
    """Сохранить результат онлайн-матча в match_log."""
    db = await _conn()
    try:
        await db.execute("INSERT INTO match_log (game, created_at, data) VALUES (?, ?, ?)",
                         (game, _now(), data_json))
        await db.commit()
    finally:
        await _release(db)


async def set_user_name(user_id: int, name: str):
    db = await _conn()
    try:
        await db.execute("UPDATE users SET display_name = ? WHERE id = ?", (name, user_id))
        await db.commit()
    finally:
        await _release(db)


async def create_google_user(username: str, email: str, google_sub: str, display_name: str = None):
    """Новый аккаунт через Google: пароля нет ('' — bcrypt его не примет), есть email/google_sub."""
    db = await _conn()
    try:
        cur = await db.execute(
            "INSERT INTO users (username, password, email, google_sub, display_name) VALUES (?, '', ?, ?, ?)",
            (username, email, google_sub, display_name or None),
        )
        user_id = cur.lastrowid
        await db.execute("INSERT INTO dictionaries (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, "default", _now()))
        await db.commit()
        return {"user_id": user_id}
    except aiosqlite.IntegrityError:
        return {"error": "User already exists"}
    finally:
        await _release(db)


async def set_user_google(user_id: int, google_sub: str, email: str):
    """Привязать Google к существующему аккаунту. IntegrityError, если этот sub уже занят."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET google_sub = ?, email = ? WHERE id = ?", (google_sub, email, user_id))
        await db.commit()
    finally:
        await _release(db)


async def clear_user_google(user_id: int):
    """Отвязать Google (email оставляем как контакт)."""
    db = await _conn()
    try:
        await db.execute("UPDATE users SET google_sub = NULL WHERE id = ?", (user_id,))
        await db.commit()
    finally:
        await _release(db)
