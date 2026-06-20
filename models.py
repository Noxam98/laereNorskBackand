from pydantic import BaseModel
from typing import Optional, List


class UserAuth(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str


class RefreshRequest(BaseModel):
    refresh_token: str


class GoogleAuth(BaseModel):
    credential: str   # ID-token (JWT) от Google Identity Services


class PasswordBody(BaseModel):
    password: str


class NameBody(BaseModel):
    name: str


class AiWordsBody(BaseModel):
    level: str = ""
    topic: str = ""
    count: int = 10
    lang: str = "ru"


class PoolEditBody(BaseModel):
    translate: dict = {}   # {no?, ru?, ukr?, en?, pl?, lt?: [варианты]}
    lang: str = "ru"       # язык интерфейса — на нём вернуть причину ревью
    hint: str = ""         # подсказка пользователя (напр. «это глагол, не существительное»)


class DictCreate(BaseModel):
    name: str


class StudyingBody(BaseModel):
    studying: bool   # включить/выключить словарь в «Учёбе»


class AddWords(BaseModel):
    prompt: str
    model: Optional[str] = None


class ImportDict(BaseModel):
    name: str
    words: List[dict] = []


class PoolAdd(BaseModel):
    norwegian: str


class PoolToDict(BaseModel):
    name: str
    q: Optional[str] = None
    topics: List[str] = []
    level: Optional[str] = None


class WordOverride(BaseModel):
    translate: Optional[dict] = None
    part_of_speech: Optional[str] = None


class ResultBody(BaseModel):
    correct: bool
    mode: Optional[str] = None       # вид игры (choice/build/input/study) — кормит SRS «Учёбы»
    elapsed: Optional[float] = None  # время ответа, сек
    direction: Optional[str] = None  # no2int | int2no — направление клетки рампы


class MoveWords(BaseModel):
    ids: List[int]
    dict_id: int


class RefineWords(BaseModel):
    ids: List[int]
    lang: str


class CurrentDictBody(BaseModel):
    name: str


class ThemeBody(BaseModel):
    theme: str


class GamePrefsBody(BaseModel):
    type: str | None = None   # study | input | choice
    dir: str | None = None    # no2int | int2no
    sound: bool | None = None


class RedescribeBody(BaseModel):
    hint: str | None = None   # подсказка о правильном значении слова


class AskBody(BaseModel):
    question: str = ""        # вопрос пользователя о слове
    lang: str = "ru"          # язык ответа


# --- «Учёба» (интервальные повторения) ---
class LearningAnswer(BaseModel):
    pool_id: int
    correct: bool
    elapsed: float | None = None
    mode: str | None = None        # choice | build | input | study | …
    direction: str | None = None   # no2int | int2no — направление клетки рампы


class PlacementBody(BaseModel):
    lang: str = "ru"
    answers: list = []        # [{no, level, answer}]


class LevelBody(BaseModel):
    level: str                # самооценка уровня (A1..C2)


class LearningStatusBody(BaseModel):
    action: str = ""          # know | reset | unarchive


class SuggestBody(BaseModel):
    count: int = 10
    level: str = ""           # A1..C2 или пусто (автоопределение)


class GateExamBody(BaseModel):
    lang: str = "ru"
    answers: list = []        # [{pool_id, answer}] — выбранный перевод для каждого вопроса


class AuditBody(BaseModel):
    lang: str = "ru"
    answers: list = []        # [{pool_id, answer}] — выбранный перевод для каждого аудит-вопроса


class RediffBody(BaseModel):
    a: str
    b: str
    lang: str = "ru"
    hint: str | None = None   # подсказка для перегенерации разницы
