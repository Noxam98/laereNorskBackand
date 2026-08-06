"""CORS для Android-приложения (Capacitor).

WebView Capacitor на Android шлёт запросы с origin `localhost` — если его нет в `CORS_ORIGINS`,
первый же запрос из APK умирает на preflight. CORS у нас fail-closed (точный список из env, без
wildcard в проде), поэтому origin приложения обязан быть в списке ЯВНО, а посторонние — не
проходить.

Схема: у Capacitor ≥3 (у нас 8.x) `server.androidScheme` по умолчанию **https**, т.е. страница
живёт на `https://localhost` — реальный Origin именно такой. `http://localhost` держим в списке
как запасной вариант (если схему когда-нибудь переключат на http) и потому, что так написано в
плане. Обе строки проверяются тестом; `capacitor://localhost` (iOS) НЕ добавляем.

Приложение собирает `allow_origins` на импорте (`main.py`), значение приходит из `config`,
который читает env на импорте. Поэтому тест перезагружает `config` + `main` с прод-подобным
списком, а в teardown возвращает исходное состояние — чтобы не влиять на другие тесты.
"""
import os
import importlib

import pytest
from fastapi.testclient import TestClient

# Строка, которую человек ставит в env прода (см. README → «CORS: origin'ы»).
PROD_LIKE_ORIGINS = "https://learnnorsk.space,https://localhost,http://localhost"
ANDROID_ORIGINS = ["https://localhost", "http://localhost"]   # WebView Capacitor (Android)
FOREIGN_ORIGIN = "https://evil.example"      # посторонний сайт — не должен проходить
REAL_PATH = "/learning/session"              # реальный эндпоинт «Учёбы»


def _reload_app(cors_value: str | None):
    """Пересобрать приложение с заданным CORS_ORIGINS (None — переменной нет вовсе)."""
    if cors_value is None:
        os.environ.pop("CORS_ORIGINS", None)
    else:
        os.environ["CORS_ORIGINS"] = cors_value
    import config
    importlib.reload(config)          # CORS_ORIGINS читается из env на импорте config
    import main
    return importlib.reload(main)     # allow_origins считается на импорте main


@pytest.fixture
def prod_like_app():
    """Приложение с fail-closed списком origin'ов, как на проде (+ origin Android-приложения)."""
    prev = os.environ.get("CORS_ORIGINS")
    main = _reload_app(PROD_LIKE_ORIGINS)
    # TestClient без `with` не запускает startup — фоновые воркеры и init_db не поднимаются.
    yield main, TestClient(main.app)
    _reload_app(prev)


def test_allow_origins_is_fail_closed_list(prod_like_app):
    """Список конкретный (не wildcard) и содержит origin Android-приложения."""
    main, _ = prod_like_app
    assert main.allow_origins != ["*"]
    for origin in ANDROID_ORIGINS:
        assert origin in main.allow_origins
    # эндпоинт из теста — реальный, а не выдуманный
    from routers.learning import router as learning_router
    assert REAL_PATH in {getattr(r, "path", None) for r in learning_router.routes}
    assert learning_router in [getattr(r, "original_router", None) for r in main.app.routes]


@pytest.mark.parametrize("origin", ANDROID_ORIGINS)
def test_preflight_from_capacitor_webview_is_allowed(prod_like_app, origin):
    """Preflight с origin WebView → приложение получает разрешение (иначе первый же
    запрос из APK умрёт, не дойдя до эндпоинта)."""
    _, client = prod_like_app
    r = client.options(REAL_PATH, headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "authorization",
    })
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == origin


@pytest.mark.parametrize("origin", ANDROID_ORIGINS)
def test_real_request_from_capacitor_webview_carries_cors_header(prod_like_app, origin):
    """И сам ответ (даже 401 без токена) несёт access-control-allow-origin — иначе WebView
    не отдаст тело/статус коду приложения."""
    _, client = prod_like_app
    r = client.get(REAL_PATH, headers={"Origin": origin})
    assert r.status_code == 401          # эндпоинт живой и защищённый (не 404)
    assert r.headers.get("access-control-allow-origin") == origin


def test_foreign_origin_still_blocked(prod_like_app):
    """Fail-closed не сломан: посторонний origin не получает разрешения ни на preflight,
    ни на обычный запрос."""
    _, client = prod_like_app
    pre = client.options(REAL_PATH, headers={
        "Origin": FOREIGN_ORIGIN,
        "Access-Control-Request-Method": "GET",
    })
    assert pre.status_code == 400
    assert "access-control-allow-origin" not in pre.headers

    r = client.get(REAL_PATH, headers={"Origin": FOREIGN_ORIGIN})
    assert "access-control-allow-origin" not in r.headers


@pytest.mark.parametrize("origin", [
    "https://localhost.evil.example",   # суффикс-обход
    "https://evil.example/localhost",   # подстрока в пути
    "http://localhost:8100",            # другой порт — другой origin (совпадение точное)
])
def test_localhost_lookalikes_are_not_allowed(prod_like_app, origin):
    """Совпадение origin'а точное: похожие на localhost строки в белый список не попадают."""
    _, client = prod_like_app
    r = client.options(REAL_PATH, headers={
        "Origin": origin,
        "Access-Control-Request-Method": "GET",
    })
    assert "access-control-allow-origin" not in r.headers
