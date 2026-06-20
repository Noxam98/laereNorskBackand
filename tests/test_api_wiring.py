"""Проводка API «Учёбы»: модули импортируются, направление доходит до SRS,
экспортируемые имена доступны из пакета db. Тесты проверяют поведение сквозь
реальный обработчик роута, а не подгоняются под внутренности."""
import importlib

import pytest

from tests.conftest import seed_user, seed_word


def test_routers_and_main_import():
    """(1) routers.learning и main импортируются без ошибок, роут /learning/answer есть."""
    learning = importlib.import_module("routers.learning")
    main = importlib.import_module("main")
    assert hasattr(learning, "router")
    # роут ответа определён на роутере «Учёбы»
    paths = {getattr(r, "path", None) for r in learning.router.routes}
    assert "/learning/answer" in paths
    # и этот роутер реально подключён в приложение (а не просто существует в модуле)
    included = [getattr(r, "original_router", None) for r in main.app.routes]
    assert learning.router in included


async def test_direction_reaches_apply_result_via_route(fresh_db):
    """(2) direction из тела запроса доходит до apply_result сквозь learning_answer:
    клетка рампы 'choice_int2no' проставляется именно для переданного направления,
    а соседняя клетка того же режима ('choice_no2int') остаётся непройденной."""
    from routers.learning import learning_answer_route
    from models import LearningAnswer
    from db.learning import get_learning

    uid, did = await seed_user()
    pid, _ = await seed_word(did, "hus", "дом")
    # первый ответ — пассивная карточка (study), чтобы появились попытки и рампа стала активной
    await learning_answer_route(LearningAnswer(pool_id=pid, correct=True, mode="study"), user={"id": uid})

    res = await learning_answer_route(
        LearningAnswer(pool_id=pid, correct=True, elapsed=1.5, mode="choice", direction="int2no"),
        user={"id": uid},
    )
    # apply_result проставил именно переданное направление, не противоположное
    assert res["modes"].get("choice_int2no") == "1"
    assert res["modes"].get("choice_no2int", "") != "1"

    # и это персистировано: видно в выборке слова
    listing = await get_learning(uid)
    w = next(x for x in listing["words"] if x["pool_id"] == pid)
    assert w["modes"].get("choice_int2no") == "1"
    assert w["modes"].get("choice_no2int", "") != "1"


async def test_wrong_direction_for_mode_is_ignored(fresh_db):
    """direction, не допустимый для режима (build осмыслен только int2no), не создаёт клетку —
    подтверждает, что значение реально проходит валидацию в apply_result, а не пишется вслепую."""
    from routers.learning import learning_answer_route
    from models import LearningAnswer

    uid, did = await seed_user()
    pid, _ = await seed_word(did, "katt", "кот")
    await learning_answer_route(LearningAnswer(pool_id=pid, correct=True, mode="study"), user={"id": uid})

    res = await learning_answer_route(
        LearningAnswer(pool_id=pid, correct=True, mode="build", direction="no2int"),
        user={"id": uid},
    )
    assert "build_no2int" not in res["modes"]
    assert "build_int2no" not in res["modes"]


def test_db_exports_available():
    """(3) экспортируемые имена «Учёбы» доступны из пакета db и вызываемы."""
    import db

    expected = [
        "learning_get", "learning_stats", "learning_due", "learning_answer",
        "learning_set_status", "learning_suggest", "learning_placement",
        "learning_grade", "learning_activity", "learning_set_level",
        "learning_seed_starter", "learning_session", "learning_gate_status",
        "learning_new_blocked", "learning_gate_exam", "learning_gate_grade",
        "learning_audit", "learning_audit_grade", "learning_audit_throttled",
    ]
    for name in expected:
        assert hasattr(db, name), f"db.{name} не экспортирован"
        assert callable(getattr(db, name)), f"db.{name} не вызываемый"

    # learning_answer — это и есть apply_result (тот же объект)
    from db.learning import apply_result
    assert db.learning_answer is apply_result
