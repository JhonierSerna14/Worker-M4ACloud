import sys

from loguru import logger

from .config import LOG_LEVEL, LOG_PATH


def configure_logging():
    logger.remove()
    if sys.stderr is not None:
        logger.add(
            sys.stderr,
            level=LOG_LEVEL,
            colorize=True,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )
    logger.add(str(LOG_PATH), rotation="10 MB", retention="7 days", level="DEBUG")
