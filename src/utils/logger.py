"""
Logger utility for the Photo Editor application.
Provides structured logging with file and console output.
"""

import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "photo_editor", log_dir: str = None) -> logging.Logger:
    """
    Set up and return a configured logger instance.

    Args:
        name: Logger name identifier.
        log_dir: Directory to store log files. Defaults to 'logs/' in the app directory.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Console handler - shows INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "[%(levelname)s] %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - logs everything including DEBUG
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"photo_editor_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    logger.info(f"Logger initialized. Log file: {log_file}")
    return logger


def get_logger(name: str = "photo_editor") -> logging.Logger:
    """
    Get an existing logger or create a new one.

    Args:
        name: Logger name identifier.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
