# Lære Norsk — бэкенд

FastAPI-бэкенд приложения для изучения норвежского: общий пул слов с ИИ-переводами,
SRS-«Учёба» (интервальные повторения с рампой), личные наборы, онлайн-игры, озвучка.
Фронтенд — в отдельном репозитории `learnNorskApp` (React/Vite → Vercel).

## Запуск

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
uvicorn main:app --reload          # API на http://127.0.0.1:8000
```

База — локальный SQLite (`users.db`, путь через `DATABASE_PATH`). LLM/TTS/пуши включаются
ключами в окружении; без них приложение работает в урезанном режиме (не падает).

## Тесты (держать зелёными)

```bash
python -m pytest -q                 # ~100 тестов SRS/сессии/экзаменов/наборов
python -m coverage run --source=db -m pytest && python -m coverage report
```

Прогоняются автоматически в **CI** (GitHub Actions, `.github/workflows/ci.yml`) на push/PR.
Тест-инфраструктура — `tests/conftest.py`: фикстура `fresh_db` (свежая БД на тест) +
хелперы `seed_user`/`seed_word`. Перед рискованным рефактором запускай `pytest` — это сеть.

## Структура

```
main.py            # сборка FastAPI-приложения + старт фоновых воркеров (lifespan)
auth.py            # JWT, регистрация/логин, Google-вход, админ-доступ
routers/           # HTTP-эндпоинты: learning, sets, pool, words, online, push…
db/                # слой данных (реэкспорт через db/__init__ → `from db import X`)
  ├── core.py          # соединение, схема (init_db), нормализация, sqlite-vec
  ├── pool.py          # общий пул слов: поиск, переводы, TTS-флаги, дедуп
  ├── dictionaries.py  # словари/наборы пользователя, слова в них
  ├── learning.py      # ЯДРО SRS: рампа, build_session, apply_result, статистика
  ├── exams.py         # экзамен-ворота (§2.4-A) + аудит забывания (§2.4-B)
  ├── placement.py     # входной тест/калибровка уровня + досев стартовых слов
  └── leaderboard.py   # рейтинг + дневная активность
autofill.py        # фоновые воркеры (LLM): добор слов, переводы, описания, формы,
                   #   эмбеддинги, дедуп, озвучка, бэкилл «ё» (yo_fix_loop)
llm/               # клиент Gemini (OpenAI-совместимый): ask_json, квоты/ключи, эмбеддинги
task.py            # системный промпт переводчика/генератора слов
```

`db/learning.py` — сердце SRS; «листовые» фичи (экзамены, плейсмент, рейтинг) вынесены
в отдельные модули и реэкспортируются из learning.py (поэтому `from db.learning import …`
и `from db import …` не меняются при дроблении).

## Деплой

```bash
flyctl deploy --remote-only        # на Fly.io
git push origin master             # + в GitHub
```
