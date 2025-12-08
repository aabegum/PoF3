"""
Centralized logging utility for PoF3 pipeline
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


LOG_DIR = "logs"


def setup_logger(step_name: str, log_dir: str = None) -> logging.Logger:
    """Setup logger for pipeline steps 01, 02, 03"""
    if log_dir is None:
        log_dir = LOG_DIR

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{step_name}_{ts}.log")

    logger = logging.getLogger(step_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def get_logger(module_name: str, log_file: Path = None) -> logging.Logger:
    """Get logger for pipeline steps 04, 05"""
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    if log_file is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(LOG_DIR) / f"{module_name}_{ts}.log"
    else:
        os.makedirs(log_file.parent, exist_ok=True)

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger
