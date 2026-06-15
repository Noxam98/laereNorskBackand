import os
import asyncio
import hashlib
from config import logger

# --- Объектное хранилище Tigris (S3-совместимое) для аудио озвучки переводов ---
# Секреты прописываются автоматически через `fly storage create` (AWS_* + BUCKET_NAME).
_BUCKET = os.getenv("BUCKET_NAME")
_ENDPOINT = os.getenv("AWS_ENDPOINT_URL_S3")
_REGION = os.getenv("AWS_REGION", "auto")

_client = None


def enabled():
    return bool(_BUCKET and _ENDPOINT and os.getenv("AWS_ACCESS_KEY_ID"))


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client(
            "s3", endpoint_url=_ENDPOINT, region_name=_REGION,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _client


def key_for(lang: str, text: str) -> str:
    h = hashlib.sha1((text or "").strip().lower().encode("utf-8")).hexdigest()
    return f"tts/{lang}/{h}.mp3"


def _get_sync(key):
    try:
        r = _get_client().get_object(Bucket=_BUCKET, Key=key)
        return r["Body"].read()
    except Exception:
        return None  # нет объекта (или ошибка чтения) — сгенерим заново


def _put_sync(key, data, content_type="audio/mpeg"):
    _get_client().put_object(Bucket=_BUCKET, Key=key, Body=data, ContentType=content_type)


# boto3 синхронный — выносим в тред, чтобы не блокировать событийный цикл.
async def get_object(key):
    if not enabled():
        return None
    return await asyncio.to_thread(_get_sync, key)


async def put_object(key, data, content_type="audio/mpeg"):
    if not enabled():
        return
    try:
        await asyncio.to_thread(_put_sync, key, data, content_type)
    except Exception as e:
        logger.warning(f"storage put '{key}': {e}")
