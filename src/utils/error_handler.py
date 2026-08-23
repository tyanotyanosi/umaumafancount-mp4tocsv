import logging
from enum import Enum
from typing import Optional


class ErrorLevel(Enum):
    USER_ERROR = "user"
    SYSTEM_ERROR = "system"
    LOG_ERROR = "log"


class ErrorHandler:
    """エラーハンドリング"""

    def __init__(self, log_file: str = "error.log"):
        self.logger = logging.getLogger("mov-to-fan-count")
        self.logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def handle(self, level: ErrorLevel, message: str, exc: Optional[Exception] = None):
        """エラー処理"""
        if level == ErrorLevel.USER_ERROR:
            self.logger.warning(f"[USER] {message}")
        elif level == ErrorLevel.SYSTEM_ERROR:
            self.logger.error(f"[SYSTEM] {message}", exc_info=exc)
        elif level == ErrorLevel.LOG_ERROR:
            self.logger.debug(f"[LOG] {message}")
