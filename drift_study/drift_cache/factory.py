from typing import Any, Dict

from drift_study.drift_cache.csv_drift_cache import CsvDriftCache
from drift_study.drift_cache.drift_cache import DriftCache
from drift_study.drift_cache.parquet_drift_cache import ParquetDriftCache
from drift_study.drift_logger.postgres_drift_logger import PostgresDriftLogger
from drift_study.exception import InvalidConfigurationException

config_caches = {
    "csv": CsvDriftCache,
    "parquet": ParquetDriftCache,
    "postgres": PostgresDriftLogger,
}


def create_drift_cache(config: Dict[str, Any]) -> DriftCache:
    """Create drift cache object from dict configuration.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration for drift cache.

    Returns
    -------
    DriftCache
        Drift Cache.

    Raises
    ------
    InvalidConfigurationException
        Raise in case of invalid configuration.
    """
    if config.get("type") is None:
        raise InvalidConfigurationException(
            f"Drift logger type not specified in {config}."
        )

    cache_type = config["type"]
    if cache_type not in config_caches:
        raise InvalidConfigurationException(
            f"Invalid drift logger type."
            f"Valid types are: {config_caches.keys()}."
        )

    logger = config_caches[cache_type](**config["params"])

    return logger
