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

### Пуши: два канала

Напоминания «N часов бездействия» уходят двумя путями, и **воркеры дедуплицируют по
пользователю** — у кого есть и веб, и приложение, получает одно уведомление:

- **веб** (`webpush.py`) — VAPID-подписка Service Worker'а;
- **Android-приложение** (`fcm.py`) — FCM HTTP v1. Web Push в Capacitor WebView не работает,
  поэтому нативу нужен свой канал.

Пороги и тихие часы у каналов общие (`PUSH_IDLE_HOURS`, `PUSH_CHECK_INTERVAL_SEC`,
`PUSH_QUIET_FROM`/`PUSH_QUIET_TO`) — намеренно, чтобы каналы не разъехались. Для FCM нужен
сервис-аккаунт Firebase (Project settings → Service accounts → Generate new private key):

```bash
FCM_SERVICE_ACCOUNT_FILE=/opt/norsk/fcm-sa.json   # рекомендуемо на VPS: chmod 600, вне git
# либо FCM_SERVICE_ACCOUNT_JSON=<весь JSON одной строкой>
# FCM_PROJECT_ID — опционально, по умолчанию project_id из JSON
```

Не задано ничего → FCM-канал выключен, бэкенд поднимается как обычно.

## CORS: origin'ы (`CORS_ORIGINS`)

CORS **fail-closed**: `main.py` берёт из env `CORS_ORIGINS` точный список origin'ов (совпадение
строгое, порт и схема — часть origin'а). `*` разрешён только для локальной разработки: с ним
креденшелы автоматически отключаются (wildcard + credentials невалиден по спеку).
**Код под это править не нужно — origin'ы меняются переменной окружения.**

- **Локально** (`.env`, в гит не коммитится): `CORS_ORIGINS=*` — покрывает и Vite
  (`http://localhost:5173`), и WebView мобильного приложения. Хочешь как на проде — поставь
  явный список (строка ниже) и перезапусти uvicorn.
- **Мобильное приложение (Capacitor, Android)**: страница живёт внутри WebView на `localhost`,
  и запросы к API уходят с origin'ом `https://localhost` (у Capacitor ≥3 `server.androidScheme`
  по умолчанию `https`; `http://localhost` держим запасным на случай смены схемы). Без этих
  origin'ов первый же запрос из APK умирает на preflight. `capacitor://localhost` — origin iOS,
  не добавляем, iOS-сборки нет.
- Тесты на это — `tests/test_cors_capacitor.py` (preflight с обоих localhost-origin'ов проходит,
  посторонний и «похожие» origin'ы — нет).

**Прод (Hetzner VPS `api.learnnorsk.space`)** — origin'ы фронта на Vercel **сохранить**, к ним
дописать два localhost-origin'а приложения:

```bash
ssh -i ~/.ssh/learnnorsk_vps -p 2222 root@204.168.163.44
grep CORS_ORIGINS /opt/norsk/.env          # посмотреть текущее значение и дописать к нему
# итоговая строка (домены фронта — те, что уже стоят; localhost-origin'ы добавляются):
# CORS_ORIGINS=https://learnnorsk.space,https://<домен-vercel>,https://localhost,http://localhost
systemctl restart norsk                    # юнит norsk.service, uvicorn на 127.0.0.1:8080
systemctl status norsk --no-pager          # должен быть active (running)
curl -s -o /dev/null -w '%{http_code}\n' https://api.learnnorsk.space/sets   # 401 = живой
curl -s -D- -o /dev/null -X OPTIONS https://api.learnnorsk.space/learning/session \
  -H 'Origin: https://localhost' -H 'Access-Control-Request-Method: GET' | grep -i allow-origin
```

Последняя команда должна вернуть `access-control-allow-origin: https://localhost` — это и есть
проверка, что приложение сможет ходить в API.

## Тесты (держать зелёными)

```bash
.venv/bin/python -m pytest -q       # ~560 тестов SRS/сессии/экзаменов/наборов
.venv/bin/python -m coverage run --source=db -m pytest && .venv/bin/python -m coverage report
```

Запускать именно `python -m pytest`: у `.venv/bin/pytest` корень репо не попадает в `sys.path`
и `conftest.py` падает на `No module named 'db'`.

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

Прод — **Hetzner VPS** `api.learnnorsk.space` (Caddy + uvicorn на 127.0.0.1:8080, юнит
`norsk.service`, код в `/opt/norsk/app`). Fly.io больше не используется.

Деплой **автоматический**: push в `master` → `.github/workflows/ci.yml` гоняет pytest и, если
зелено, сам катит прод (rsync кода → `backup.sh` → `uv pip install` → `systemctl restart norsk`
→ health-check до 2 минут).

```bash
git push origin master             # это и есть деплой, отдельной команды нет
```

Ручной путь (если CI недоступен):

```bash
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ \
  --exclude 'users.db*' \
  -e "ssh -p 2222" ./ root@204.168.163.44:/opt/norsk/app/
ssh -p 2222 root@204.168.163.44 '/opt/norsk/backup.sh && systemctl restart norsk'
curl -s -o /dev/null -w '%{http_code}\n' https://api.learnnorsk.space/sets   # 401 = живой
```

`fly.toml` и `deploy.sh` — артефакты старого хостинга, приложение на Fly удалено.

## Данные и лицензии

Словарная база собирается из открытых источников; полный конвейер сборки — офлайн
(харвест), обогащение — фоновые воркеры.

| Источник | Что берём | Лицензия |
|---|---|---|
| [Norsk ordbank](https://www.nb.no/sprakbanken/ressurskatalog/oai-nb-no-sbr-5/) (Språkbanken) | грамматические формы, род | CC BY 4.0 |
| [Bokmålsordboka](https://ord.uib.no) (UiB/Språkrådet, живой API) | формы новых слов (nyord) | открытые данные, с указанием источника |
| LEXIN (OsloMet / HK-dir) | человеческие переводы, примеры | открытые данные, с указанием источника |
| [OpenSubtitles-частоты](https://github.com/hermitdave/FrequencyWords) | разговорные частоты для уровней | CC BY-SA 4.0 |

**Уровни CEFR** слов считаются собственной методикой без проприетарных словников:
`min(полоса по книжному корпусу, полоса по субтитровому корпусу, LLM-оценка уровня
учебника)`. LLM-оценка («на каком уровне учебников слово вводится») используется потому,
что существующие учебные частотные списки для норвежского (например, Kelly-list)
распространяются под некоммерческими лицензиями (CC BY-NC-SA) и в отгружаемые данные
приложения не входят — такие списки применяются только как внутренний эталон качества
калибровки. Итоговый уровневый словник опубликован в этом репозитории — `data/levels-v1.json`
(CC BY-SA 4.0, наследование от субтитровых частот).
