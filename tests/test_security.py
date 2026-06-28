"""Безопасность: SSRF-валидация push-endpoint (allowlist) и rate-limit-логика (чистые функции)."""
import pytest
from fastapi import HTTPException

from webpush import _valid_push_endpoint
from ratelimit import _hit


def test_push_endpoint_allowlist_blocks_ssrf():
    # легитимные push-сервисы
    assert _valid_push_endpoint("https://fcm.googleapis.com/fcm/send/abc")
    assert _valid_push_endpoint("https://updates.push.services.mozilla.com/wpush/v2/xyz")
    assert _valid_push_endpoint("https://web.push.apple.com/abc")
    # SSRF / мусор / обходы — отвергаем
    assert not _valid_push_endpoint("http://fcm.googleapis.com/x")             # не https
    assert not _valid_push_endpoint("https://169.254.169.254/latest/meta")     # internal IP
    assert not _valid_push_endpoint("https://localhost:8000/x")
    assert not _valid_push_endpoint("https://evil.com/x")
    assert not _valid_push_endpoint("https://fcm.googleapis.com.evil.com/x")   # суффикс-обход
    assert not _valid_push_endpoint("https://evil-googleapis.com/x")           # без точки-границы
    assert not _valid_push_endpoint("")
    assert not _valid_push_endpoint(None)


def test_rate_limit_blocks_after_max():
    key = ("test", "u-rl")
    for _ in range(3):
        _hit(key, 3, 60)                 # 3 разрешены
    with pytest.raises(HTTPException) as e:
        _hit(key, 3, 60)                 # 4-й → 429
    assert e.value.status_code == 429
    assert "Retry-After" in (e.value.headers or {})
