import os
from datetime import datetime
from rich.logging import RichHandler
import logging

def setup_logger(log_dir):
    """
    Setup a colorful, nicely formatted logger using Rich.

    Logs go both to a timestamped file and to the console with color highlighting.
    """
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(log_dir, f"run_{dt}.log")

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler (plain text, no colors)
    file_handler = logging.FileHandler(file_path)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler with Rich colors
    console_handler = RichHandler(rich_tracebacks=True)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    return logger
