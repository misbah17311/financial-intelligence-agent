"""
Logging setup using loguru.
We log queries, retrieved docs, agent decisions — basically the whole
pipeline trace so we can debug and evaluate.
"""

import sys
from loguru import logger
from src.config import LOG_LEVEL, ROOT_DIR

# remove default handler and set our own format
logger.remove()
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
)

# also dump to a log file for full trace
logger.add(
    ROOT_DIR / "agent.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} | {message}",
)
