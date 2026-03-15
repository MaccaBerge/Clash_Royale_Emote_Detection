import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform

from config.constants import Constants


def get_log_file_path(app_name: str) -> Path:
    system = platform.system().lower()

    app_support_dir = Path(
        os.path.expanduser(f"{Constants.path.app_help_dir[system]}/{app_name}")
    )
    app_support_dir.mkdir(parents=True, exist_ok=True)

    return app_support_dir / Constants.logger.log_file_name


program_name = Constants.general.program_name
logger_name = Constants.logger.app_logger_name

logger = logging.getLogger(logger_name)
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    log_file = get_log_file_path(program_name)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=Constants.logger.log_file_max_bytes,
        backupCount=Constants.logger.log_file_backup_count,
        encoding=Constants.logger.log_file_encoding,
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        Constants.logger.app_logger_format,
        datefmt=Constants.logger.app_logger_date_format,
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
