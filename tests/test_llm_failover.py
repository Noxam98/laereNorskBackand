"""D1: отказоустойчивость LLM-слоя — самая частая точка отказа в проде, раньше без тестов.
Покрываем ЧИСТУЮ логику (без сети): quota.candidates (round-robin + кулдаун 429), errors.classify
(ветки по статусу/типу), errors.quota_info (парс retryDelay), errors.report (не пробрасывает —
на этом держится «цикл переживает падение итерации», D3)."""
from llm import quota
import errors


def _reset():
    quota._blocked.clear()
    quota._rr_cursor.clear()


# ---------------- quota.candidates: порядок ключей ----------------
def test_candidates_round_robin():
    _reset()
    order = quota.candidates(["m"], ["a", "b", "c"], "text")
    assert [i for _, _, i in order] == [0, 1, 2]           # с чистого курсора — по порядку
    quota.advance("text", "m", 1)                          # успешно отработал ключ 1
    order = quota.candidates(["m"], ["a", "b", "c"], "text")
    assert [i for _, _, i in order] == [2, 0, 1]           # следующий старт — после ключа 1


def test_candidates_429_goes_to_tail():
    _reset()
    quota.mark_429("text", "m", 0, seconds=1000)           # ключ 0 в кулдауне надолго
    order = quota.candidates(["m"], ["a", "b", "c"], "text")
    ids = [i for _, _, i in order]
    assert ids[-1] == 0 and set(ids) == {0, 1, 2}          # «остывающий» — в хвосте, но не выпал
    assert not quota._fresh("text", "m", 0)                # он реально помечен несвежим
    assert quota._fresh("text", "m", 1)


def test_available_when_some_key_fresh():
    _reset()
    keys = ["a", "b"]
    assert quota._any_fresh(["m"], keys, "text") is True
    quota.mark_429("text", "m", 0, seconds=1000)
    quota.mark_429("text", "m", 1, seconds=1000)
    assert quota._any_fresh(["m"], keys, "text") is False  # все в кулдауне → нечем ходить


# ---------------- errors.classify: ветки ----------------
class _Exc(Exception):
    def __init__(self, msg="", status=None):
        super().__init__(msg)
        self.status_code = status


def test_classify_by_status():
    assert errors.classify(_Exc(status=429)).kind == errors.QUOTA
    assert errors.classify(_Exc(status=429)).http_status == 429
    assert errors.classify(_Exc(status=401)).kind == errors.AUTH
    assert errors.classify(_Exc(status=403)).kind == errors.AUTH
    assert errors.classify(_Exc(status=503)).kind == errors.SERVER
    assert errors.classify(_Exc(status=400)).kind == errors.BAD_REQUEST
    # 400 — наша ошибка (напр. схема), НЕ алёртим и не инцидент
    assert errors.classify(_Exc(status=400)).alert is False


def test_classify_quota_by_text_and_internal():
    assert errors.classify(_Exc("RESOURCE_EXHAUSTED: ...")).kind == errors.QUOTA
    # без статуса и провайдерских маркеров — это НАШ внутренний сбой, не Gemini
    assert errors.classify(RuntimeError("sqlite is locked")).kind == errors.INTERNAL


def test_quota_info_retry_delay():
    e1 = _Exc("x")
    e1.body = {"error": {"details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo",
                                      "retryDelay": "42s"}]}}
    assert quota_info_retry(e1) == 42.0
    # фолбэк из текста
    assert quota_info_retry(_Exc("Please retry in 12.5s")) == 12.5
    assert quota_info_retry(_Exc("no hint")) is None


def quota_info_retry(exc):
    return errors.quota_info(exc)["retry"]


# ---------------- errors.report: не пробрасывает (D3 — устойчивость лупов) ----------------
def test_report_never_raises(monkeypatch):
    monkeypatch.setattr("notify.notify", lambda *a, **k: None)   # без Telegram-сети
    info = errors.report(RuntimeError("boom"), "some_loop")      # ровно то, что делают while-лупы
    assert info.kind == errors.INTERNAL                          # классифицировано, исключение проглочено
