# AGENTS.md

Инструкции для ИИ-агентов по бэкенду **Lære Norsk** (FastAPI + aiosqlite + sqlite-vec, LLM Gemini).
Обзор, запуск и структура — в [README.md](README.md). Фронт — отдельный репозиторий `learnNorskApp`.
Этот файл — про **рабочий процесс агента**: как менять код, не сломав прод.

---

## Команды (всегда через `.venv`)

В корне есть готовый `.venv` со всеми зависимостями (pytest, coverage, aiosqlite, …).

```bash
.venv/bin/python -m pytest -q                      # ~560 тестов — СЕТЬ под рефактор
.venv/bin/python -c "import main"                  # ловит циклические импорты / NameError при загрузке
.venv/bin/python -m coverage run --source=db -m pytest && .venv/bin/python -m coverage report
```

Именно `python -m pytest`: `.venv/bin/pytest` не кладёт корень репо в `sys.path`, и `conftest.py`
падает на `No module named 'db'`.

---

## Цепочка верификации (перед «готово»)

Бэкенд под прод — связывающее правило **«ничего не поломав»**. Любую правку прогоняй по цепочке:

1. **`import main`** — ловит циклические импорты и `NameError` на загрузке модулей.
2. **`pytest`** — ловит ошибки в телах ПОКРЫТЫХ функций (~560 тестов).
3. **Push = деплой** — `git push origin master`. CI (`.github/workflows/ci.yml`) гоняет pytest и
   при зелёном сам катит прод на VPS: rsync → `backup.sh` → `uv pip install` →
   `systemctl restart norsk` → health-check. Отдельной команды деплоя нет; `flyctl` мёртв.
4. **Health** — защищённые эндпоинты отвечают `401` (живой и сервит):
   `curl -s -o /dev/null -w '%{http_code}' https://api.learnnorsk.space/sets` → `401`
   (так же `/learning/session`). WS-маршрут `/ws/online` на обычный GET даёт `404` — это норма.

CI гоняет pytest на Python 3.13 на push/PR; деплой-job — только на push в `master`.

---

## Паттерн дробления `db/` — «реэкспорт снизу»

`db/` — слой данных, наружу плоский: `from db import X` / `from db.<module> import X`. Крупные
модули расщеплены на «листовые» так, чтобы **импорты НЕ менялись**:

- Листовой модуль (напр. `db/exams.py`) импортирует общие хелперы из `db.learning`/`db.core`.
- Родитель (`db/learning.py`) **в самом низу** реэкспортирует функции листового модуля:
  ```python
  from .exams import (gate_status, build_gate_exam, ...)  # noqa: E402,F401
  ```
  → `from db.learning import gate_status` и `from db import gate_status` продолжают работать.
- Цикла нет: листовой модуль импортирует только то, что объявлено в родителе ВЫШЕ блока реэкспорта.

Так уже вынесены: из `learning.py` → `leaderboard/placement/exams`; из `pool.py` →
`pool_moderation/pool_freq/pool_dedup/pool_queues`; из `dictionaries.py` → `sets_data`;
из `autofill.py` → `autofill_wordgen/autofill_enrich`; из `online.py` → `online_logic`.

Для **разбросанных** (не подряд) функций — извлечение по AST (диапазоны строк), новый модуль
импортирует зависимости напрямую из `db`/`llm` (не из родителя — чтобы без цикла), диапазоны
удаляются из родителя, реэкспорт вставляется после блока импортов.

---

## Тесты

- `tests/conftest.py`: фикстура **`fresh_db`** (свежая БД на каждый тест) + хелперы
  `seed_user()` / `seed_word(dict_id, no, ru, pos, level)`. `asyncio_mode=auto` (async-тесты без декоратора).
- Окружение ставится **до** импорта БД: `DB_POOL_SIZE=0` (соединения закрываются на release),
  `NB_LEXICON_SEED=0` (не сидим 318k-лексикон), `DATABASE_PATH=<tempfile>`.
- LLM-воркеры тестируются с **замоканным `ask_json`** — патчить во ВСЕХ модулях, где он
  используется (напр. `autofill`, `autofill_wordgen`, `autofill_enrich`), иначе перенесённая
  функция дёрнет немокнутый клиент. Пример: `tests/test_autofill.py`.
- Перед дроблением крупного файла **сначала измерь покрытие** трогаемых функций и добей тестами —
  pytest ловит ошибки только в ПОКРЫТЫХ телах.

---

## Конвенции

- **Комментарии — на русском**, «зачем», не «что». Повторяй окружающий стиль.
- **Async везде**: aiosqlite, `await _conn()` / `await _release(db)` (см. `db/core.py`).
  Векторный ANN — sqlite-vec (`vec_upsert`/`vec_nearest_rows`).
- **LLM** — только через `llm`-клиент (`ask_json`, `embed_texts`): перебор 5 Gemini-ключей на 429,
  квоты, OpenAI-совместимый формат. Vision/OCR — массив `content` с `image_url`.
- **Омонимы**: запись пула = `(norwegian, pos)`. Писать результат строго по `id` из выборки, не
  `get_pool_id(word)` без `pos` — иначе попадёшь в старшую запись и сольёшь квоту (см. memory
  `autofill-homograph-requeue`).
- **Роутеры** в `routers/` тонкие — бизнес-логика в `db.*`/`autofill`.
- **Фоновые воркеры** стартуют в `main.py` (lifespan); поодиночке — циклы в `autofill.py`,
  `tts.py`, `webpush.py`.

---

## Деплой и git

- **Деплой**: `git push origin master` — этого достаточно, CI сам катит прод на VPS.
  Только по явной просьбе пользователя (push в `master` = выкат, а не «просто сохранить»).
- **Формат коммитов**: conventional на русском (`feat`/`fix`/`refactor(pool)`/`chore`). Кратко.
- **Не коммитить**: `users.db` и прочие локальные БД, секреты. Стейджить точечно, не `git add -A`.
- **RAM на VPS** — 3.7 ГБ и **без swap** (на сервере со-хостится второй проект): тяжёлое
  считать локально, а не на инстансе. Резидентный кеш эмбеддингов (`embcache.py`) уже держит
  ~72 МБ — это основной потребитель.
