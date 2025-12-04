import logging
import os
from datetime import datetime

LOG_ROOT = 'logs'

def get_logger(module_name):
    logger = logging.getLogger(module_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(LOG_ROOT, session)
    os.makedirs(session_dir, exist_ok=True)

    logfile = os.path.join(session_dir, f"{module_name}.log")

    handler = logging.FileHandler(logfile, encoding='utf-8')
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    return logger
