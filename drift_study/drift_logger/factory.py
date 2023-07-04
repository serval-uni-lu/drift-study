from typing import Any, Dict

from drift_study.drift_logger.csv_drift_logger import CsvDriftLogger
from drift_study.drift_logger.drift_logger import DriftLogger
from drift_study.drift_logger.parquet_drift_logger import ParquetDriftLogger
from drift_study.drift_logger.postgres_drift_logger import PostgresDriftLogger
from drift_study.exception import InvalidConfigurationException

config_loggers = {
    "csv": CsvDriftLogger,
    "parquet": ParquetDriftLogger,
    "postgres": PostgresDriftLogger,
}


def create_drift_logger(config: Dict[str, Any]) -> DriftLogger:
    """Create drift logger object from dict configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration for drift logger.

    Returns
    -------
    DriftLogger
        Drift logger.

    Raises
    ------
    InvalidConfigurationException
        Raise in case of invalid configuration.
    """
    if config.get("type") is None:
        raise InvalidConfigurationException(
            f"Drift logger type not specified in {config}."
        )

    logger_type = config["type"]
    if logger_type not in config_loggers:
        raise InvalidConfigurationException(
            f"Invalid drift logger type. "
            f"Valid types are: {config_loggers.keys()}."
        )

    logger = config_loggers[logger_type](**config["params"])

    return logger
