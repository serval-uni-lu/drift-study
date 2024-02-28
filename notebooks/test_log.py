import logging


def test_log():
    logger = logging.getLogger(__name__)
    logger.info("test")
