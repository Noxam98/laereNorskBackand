import time

# Активность = последний ПОЛЬЗОВАТЕЛЬСКИЙ вызов Gemini. Фон ждёт простоя по ней.
# Изначально считаем, что простой уже наступил (после рестарта фон стартует сразу;
# 5-минутная пауза включается только после реальной активности пользователя).
_last_activity = time.monotonic() - 86400


def mark_activity():
    global _last_activity
    _last_activity = time.monotonic()


def seconds_idle():
    return time.monotonic() - _last_activity
