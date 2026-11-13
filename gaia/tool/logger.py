import logging
import sys
from functools import lru_cache
from typing import Literal


@lru_cache
def bootstrap_logger(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int = "DEBUG",
    logger: logging.Logger | None = None,
) -> logging.Logger:
    """Configure only this logger, not the root logger"""
    if logger is None:
        logger = logging.getLogger("gaia-agent")
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger
