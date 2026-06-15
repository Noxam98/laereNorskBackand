import os
import logging
from logging.handlers import RotatingFileHandler

# Единый логгер приложения (общий для всех модулей).
logger = logging.getLogger("learnnorsk")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
if not logger.handlers:  # защита от дублирования хендлеров при повторном импорте
    _fh = RotatingFileHandler('app.log', maxBytes=5 * 1024 * 1024, backupCount=3); _fh.setFormatter(_fmt); logger.addHandler(_fh)
    _sh = logging.StreamHandler(); _sh.setFormatter(_fmt); logger.addHandler(_sh)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
