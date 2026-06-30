"""Рантайм-флаги паузы фоновых задач (управляются с админ-страницы статистики).
True = поставлено на паузу. Персистятся в БД (app_settings, key='paused'),
поэтому состояние переживает рестарт/передеплой."""
import json

PAUSED = {
    "autofill": False,   # автодобавление новых слов (autofill_loop)
    "embed": False,      # фоновые эмбеддинги (reembed_loop + досчёт в autofill)
    "describe": False,   # генерация описаний (describe_loop)
    "forms": False,      # грамматические формы по части речи (forms_loop)
    "pos": False,        # переразметка части речи у «прочее» (pos_loop)
    "dedup": False,      # фоновый дедуп пула: слияние слов-дублей (dedup_loop)
    "homograph": False,  # разбиение омонимов сущ./глаг./прил. на per-pos записи (homograph_loop)
}

_SETTING_KEY = "paused"


async def load_persisted():
    """Восстановить состояние пауз из БД (вызывать на старте после init_db)."""
    from db import get_setting
    raw = await get_setting(_SETTING_KEY)
    if not raw:
        return
    try:
        saved = json.loads(raw)
    except Exception:
        return
    for k in PAUSED:
        if k in saved:
            PAUSED[k] = bool(saved[k])


async def persist():
    """Сохранить текущее состояние пауз в БД (вызывать после изменения PAUSED)."""
    from db import set_setting
    await set_setting(_SETTING_KEY, json.dumps(PAUSED))
