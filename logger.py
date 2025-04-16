import logging
import os
from datetime import datetime

LOGS_DIR = 'logs'

DATE_FMT = '%Y-%m-%d %H:%M:%S'

FMT = '%(asctime)s.%(msecs)03d - %(levelname)01s - %(threadName)s - %(message)s'

LOG_LEVEL = logging.DEBUG


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'CRITICAL': '\033[95m',  # Magenta
    }

    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)
        self.fmt = fmt
        self.date_fmt = datefmt
        self.style = style
        self.validate = validate

    def format(self, record):
        log_fmt = f"{self.COLORS.get(record.levelname, self.RESET)}{self.fmt}{self.RESET}"
        formatter = logging.Formatter(log_fmt, datefmt=self.date_fmt, style=self.style, validate=self.validate)
        return formatter.format(record)


class HiLogger:
    def __init__(self, name, log_file_prefix='app'):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)

        # 创建日志输出目录
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
        # 获取当前时间并格式化
        now = datetime.now()
        log_file = now.strftime(f"{log_file_prefix}_%Y-%m-%d_%H-%M-%S.log")
        # 创建文件处理器
        file_handler = logging.FileHandler(os.path.join(LOGS_DIR, log_file))
        file_handler.setLevel(LOG_LEVEL)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)

        # 创建日志格式
        file_handler.setFormatter(logging.Formatter(fmt=FMT, datefmt=DATE_FMT))
        console_handler.setFormatter(ColoredFormatter(fmt=FMT, datefmt=DATE_FMT))

        # 添加处理器到 logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
