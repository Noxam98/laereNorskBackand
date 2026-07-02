from pydantic import BaseModel, Field
from typing import Optional, List

# Лимиты полей — defense-in-depth: не пускаем гигантский фритекст в LLM-промпты и огромные списки
# в SQL/батчи. Значения щедрые (нормальный ввод не задевают), бьют только по абьюзу/ошибкам.


class UserAuth(BaseModel):
    username: str = Field(max_length=64)
    password: str = Field(max_length=200)


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RefreshRequest(BaseModel):
    refresh_token: str = Field(max_length=4096)


class GoogleAuth(BaseModel):
    credential: str = Field(max_length=8192)   # ID-token (JWT) от Google Identity Services


class PasswordBody(BaseModel):
    password: str = Field(max_length=200)


class NameBody(BaseModel):
    name: str = Field(max_length=80)


class AiWordsBody(BaseModel):
    level: str = Field("", max_length=8)
    topic: str = Field("", max_length=120)
    count: int = Field(10, ge=1, le=50)
    lang: str = Field("ru", max_length=8)


class PoolEditBody(BaseModel):
    translate: dict = {}   # {no?, ru?, ukr?, en?, pl?, lt?: [варианты]}
    lang: str = Field("ru", max_length=8)   # язык интерфейса — на нём вернуть причину ревью
    hint: str = Field("", max_length=500)   # подсказка пользователя (напр. «это глагол, не существительное»)


class DictCreate(BaseModel):
    name: str = Field(max_length=80)


class StudyingBody(BaseModel):
    studying: bool   # включить/выключить словарь в «Учёбе»


class AddWords(BaseModel):
    prompt: str = Field(max_length=500)
    model: Optional[str] = Field(None, max_length=64)


class ImportDict(BaseModel):
    name: str = Field(max_length=80)
    words: List[dict] = Field(default=[], max_length=500)


class PoolAdd(BaseModel):
    norwegian: str = Field(max_length=80)


class PoolToDict(BaseModel):
    name: str = Field(max_length=80)
    q: Optional[str] = Field(None, max_length=120)
    topics: List[str] = Field(default=[], max_length=64)
    level: Optional[str] = Field(None, max_length=8)


class WordOverride(BaseModel):
    translate: Optional[dict] = None
    part_of_speech: Optional[str] = Field(None, max_length=32)


class ResultBody(BaseModel):
    correct: bool
    mode: Optional[str] = Field(None, max_length=32)       # вид игры (choice/build/input/study) — кормит SRS «Учёбы»
    elapsed: Optional[float] = None  # время ответа, сек
    direction: Optional[str] = Field(None, max_length=16)  # no2int | int2no — направление клетки рампы


class MoveWords(BaseModel):
    ids: List[int] = Field(max_length=2000)
    dict_id: int


class RefineWords(BaseModel):
    ids: List[int] = Field(max_length=2000)
    lang: str = Field(max_length=8)


class CurrentDictBody(BaseModel):
    name: str = Field(max_length=80)


class ThemeBody(BaseModel):
    theme: str = Field(max_length=32)


class GamePrefsBody(BaseModel):
    type: str | None = None   # study | input | choice
    dir: str | None = None    # no2int | int2no
    sound: bool | None = None
    kbdHintSeen: bool | None = None   # видел сноску «можно печатать с клавиатуры» (ПК) — больше не показывать
    choiceHintSeen: bool | None = None  # видел сноску «можно выбирать цифрами» в «Выборе» (ПК)
    leaderboardOptOut: bool | None = None  # не участвовать в рейтинге (скрыт от других)
    listenOff: bool | None = None  # задания «на слух» выключены для аккаунта (синк между устройствами)
    lang: str | None = Field(None, max_length=8)  # язык интерфейса (синк между устройствами): ru|ukr|en|pl|lt
    newPerSession: int | None = Field(None, ge=1, le=20)  # сколько НОВЫХ слов-карточек за сессию (порционность)
    grammar: bool | None = None   # грамм-overlay (★): род/формы поверх выученных слов — вкл/выкл (мастер)
    grammarPos: dict | None = None   # пер-POS тумблеры грамматики: {noun,verb,adjective,pronoun: bool}
    audio: bool | None = None   # аудиозадания: ВКЛ → choice_no2int в отдельную слуховую сессию; ВЫКЛ → текстом в рампе
    listenPack: int | None = Field(None, ge=5, le=20)   # порог слуховой партии (сколько «ждёт слух» собрать)
    studyOnboarded: bool | None = None   # приветственное окно Учёбы пройдено (показывать однократно)


class RedescribeBody(BaseModel):
    hint: str | None = Field(None, max_length=500)   # подсказка о правильном значении слова


class AskBody(BaseModel):
    question: str = Field("", max_length=500)        # вопрос пользователя о слове
    lang: str = Field("ru", max_length=8)            # язык ответа


# --- «Учёба» (интервальные повторения) ---
class LearningAnswer(BaseModel):
    pool_id: int
    correct: bool
    elapsed: float | None = None
    mode: str | None = Field(None, max_length=32)        # choice | build | input | study | …
    direction: str | None = Field(None, max_length=16)   # no2int | int2no — направление клетки рампы
    form: bool = False                                   # трек ФОРМ: ответ идёт в form_srs (не в base/overlay)
    cell: str | None = Field(None, max_length=16)        # клетка формы (past | def_sg | gender | …)
    stage: str | None = Field(None, max_length=8)        # ступень рампы форм (card | choose | produce)


class PlacementBody(BaseModel):
    lang: str = Field("ru", max_length=8)
    answers: list = Field(default=[], max_length=200)        # [{no, level, answer}]


class LevelBody(BaseModel):
    level: str = Field(max_length=8)                # самооценка уровня (A1..C2)


class LearningStatusBody(BaseModel):
    action: str = Field("", max_length=32)          # know | reset | unarchive


class GateExamBody(BaseModel):
    lang: str = Field("ru", max_length=8)
    answers: list = Field(default=[], max_length=200)        # [{pool_id, answer}] — выбранный перевод для каждого вопроса


class AuditBody(BaseModel):
    lang: str = Field("ru", max_length=8)
    answers: list = Field(default=[], max_length=200)        # [{pool_id, answer}] — выбранный перевод для каждого аудит-вопроса


class RediffBody(BaseModel):
    a: str = Field(max_length=80)
    b: str = Field(max_length=80)
    lang: str = Field("ru", max_length=8)
    hint: str | None = Field(None, max_length=500)   # подсказка для перегенерации разницы
