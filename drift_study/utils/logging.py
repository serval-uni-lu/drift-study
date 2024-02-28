import logging.config
import warnings
from typing import Any, Dict, Optional

from optuna.exceptions import ExperimentalWarning

DEFAULT_CONFIG = {
    "version": 1,
    "formatters": {
        "simple": {
            "format": "[%(levelname)-.4s] %(asctime)s - %(name)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        }
    },
    "loggers": {
        "__main__": {"level": "DEBUG"},
        "optuna": {"level": "ERROR"},
        "matplotlib": {"level": "ERROR"},
    },
    "root": {"level": "DEBUG", "handlers": ["console"]},
}


def configure_logger(config: Optional[Dict[str, Any]] = None):
    if config is None:
        config = {}
    logging_config = config.get("logging", DEFAULT_CONFIG)
    logging.config.dictConfig(logging_config)
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
