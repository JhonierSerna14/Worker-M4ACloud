import sys

from loguru import logger

from .config import LOG_LEVEL


def configure_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add("worker.log", rotation="10 MB", retention="7 days", level="DEBUG")
