from __future__ import annotations
from loguru import logger
from .paths import logs_dir

def configure_logging() -> None:
    logger.remove()
    log_path = logs_dir() / "toolbox.log"
    logger.add(str(log_path), rotation="5 MB", retention=10, enqueue=True, backtrace=False, diagnose=False)
    logger.add(lambda msg: print(msg, end=""), level="INFO")  # console
