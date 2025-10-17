"""
Custom logger class to add line numbers to log messages with a level of WARNING or higher.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

import coloredlogs


class Logger:
    """
    Custom logger class to add line
    numbers to log messages with a level
    of WARNING or higher.

    Attributes:
        level (str): The logging level for the logger.
        fmt (str): The format for the log messages.
        log_file (str): The log file to write the log messages to.
        log_dir (str): The directory to store the log files.

    """

    def __init__(self, level=logging.INFO, fmt=None, log_file=None, log_dir="log"):
        """
        Initialize the CustomLogger class.
        """
        if fmt is None:
            fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

        self.level = level
        self.fmt = fmt
        self.log_dir = log_dir

        # Ensure the log directory exists
        if not Path(log_dir).exists():
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        if log_file is None:
            start_time = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"log{start_time}.log"

        self.log_file = str(Path(log_dir) / log_file)

    class LineNumberFilter(logging.Filter):
        """
        Custom filter class to add line numbers.
        """

        def filter(self, record):
            """Filter log records to add line number information."""
            if record.levelno >= logging.WARNING:
                record.msg = f"{record.msg} ({record.filename}:{record.lineno})"
            return True

    def get_logger(self, name):
        """
        Create and configure a logger with the provided name.

        Args:
            name (str): The name for the logger, typically use __name__ from the calling module.

        Returns:
            logging.Logger: Configured logger instance.

        """
        # Create a logger with the provided name
        logger = logging.getLogger(name)
        logger.propagate = False

        # Install coloredlogs with the base format
        coloredlogs.install(level=self.level, logger=logger, fmt=self.fmt)

        # Create and add the custom filter to the logger
        line_number_filter = self.LineNumberFilter()
        for handler in logger.handlers:
            handler.addFilter(line_number_filter)

        # Add a file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.level)
        file_handler.setFormatter(logging.Formatter(self.fmt))
        file_handler.addFilter(line_number_filter)
        logger.addHandler(file_handler)

        return logger
