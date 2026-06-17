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


class DictCreate(BaseModel):
    name: str


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


class RediffBody(BaseModel):
    a: str
    b: str
    lang: str = "ru"
    hint: str | None = None   # подсказка для перегенерации разницы
