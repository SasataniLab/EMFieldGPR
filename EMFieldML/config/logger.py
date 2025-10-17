"""Logger configuration for EMFieldML electromagnetic field toolkit."""

import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a configured logger for the given name.

    Args:
        name: The name for the logger.
        level: Optional logging level override.

    Returns:
        Configured logger instance.

    """
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    # Only add handler if none exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
