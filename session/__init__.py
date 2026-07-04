"""Сборка сессии как конвейер чистых шагов (Этапы 4-8 decoupling-плана).

Каждый шаг — функция от простых данных (enriched-элементы: dict с row/status/due/data),
без SQL и asyncio; побочные эффекты и БД остаются в оркестраторе db/learning.build_session.
"""
from . import pools, forms_phase, shape, distractors, reason, compounds  # noqa: F401
