import logging
from logging.handlers import RotatingFileHandler

class Logger:
    def __init__(self, log_file='log.txt', max_bytes=10 * 1024 * 1024, backup_count=5, level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level)

        file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(module)s.%(funcName)s(%(lineno)d) | %(message)s'))
        self.logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self.logger.addHandler(stream_handler)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)