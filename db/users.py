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


async def get_user_audio(user_id: int):
    """Аудиозадания (слуховые сессии): (включены?, порог партии 5..20). gamePrefs.audio (дефолт — вкл;
    бэк-совместимость: старый listenOff=true → выкл) + gamePrefs.listenPack (клампим 5..20, дефолт 10).
    audio ВКЛ → choice_no2int откладывается в слуховую сессию; ВЫКЛ → идёт в дневной рампе текстом."""
    db = await _conn()
    try:
        async with db.execute("SELECT game_prefs FROM users WHERE id = ?", (user_id,)) as cur:
            row = await cur.fetchone()
    finally:
        await _release(db)
    audio_on, pack = True, 10
    if row and row["game_prefs"]:
        try:
            gp = json.loads(row["game_prefs"])
            v = gp.get("audio")
            if isinstance(v, bool):
                audio_on = v
            elif gp.get("listenOff") is True:   # старый тумблер «на слух выключен» → аудио выкл
                audio_on = False
            p = gp.get("listenPack")
            if isinstance(p, (int, float)):
                pack = max(5, min(20, int(p)))
        except Exception:
            pass
    return audio_on, pack


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
