import logging
import sys
import os

DEFAULT_LOGGER_NAME = os.getenv('LOGGER_NAME', 'cv2pipeline')


def init_logging(logger_name=DEFAULT_LOGGER_NAME,
                 log_level=logging.DEBUG,
                 format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                 stream=None):
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(format)
    if stream is None:
        stream = sys.stdout
    ch = logging.StreamHandler(stream=stream)
    ch.setLevel(log_level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)

    return logger


def get_default_logger():
    return logging.getLogger(DEFAULT_LOGGER_NAME)
