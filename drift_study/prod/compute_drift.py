import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from configutils import get_config
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset_from_config
from sqlalchemy import create_engine

from drift_study.drift_cache.drift_cache import DriftCache
from drift_study.drift_cache.factory import create_drift_cache
from drift_study.drift_detectors.data_based.divergence_drift import (
    DivergenceDrift,
)
from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.drift_logger.drift_logger import DriftLogger
from drift_study.drift_logger.factory import create_drift_logger
from drift_study.drift_logger.multi_logger import MultiDriftLogger
from drift_study.drift_logger.static_time_logger import StaticTimeDriftLogger


def get_logger(
    logger_config: List[Dict[str, Any]], current_time: int
) -> DriftLogger:
    loggers = [create_drift_logger(c) for c in logger_config]
    logger = StaticTimeDriftLogger(MultiDriftLogger(loggers), current_time)
    return logger


def get_test(dataset: Dataset) -> pd.DataFrame:
    N_TRAIN = 400000
    N_TEST = 300000
    x, _ = dataset.get_x_y()
    x_test = x.iloc[N_TRAIN : N_TRAIN + N_TEST]
    return x_test


def get_db_test(current_time, time_window) -> pd.DataFrame:
    USERNAME = "postgres"
    PASSWORD = "example"
    DATABASE = "lcld"
    SERVER = "localhost"

    engine = create_engine(
        f"postgresql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    )

    query = (
        f"SELECT * FROM x_data "
        f"WHERE date <= to_timestamp({current_time}) "
        f"AND date >= to_timestamp({current_time - time_window}) "
        f"ORDER BY date"
    )

    return pd.read_sql_query(query, engine).drop(columns=["date", "index"])


def get_drift_detector(
    dataset: Dataset, cache: DriftCache, logger: DriftLogger
) -> DriftDetector:
    x_metadata = dataset.get_metadata(only_x=True)
    detector = DivergenceDrift(
        x_metadata=x_metadata,
        numerical_features=x_metadata["feature"][
            (x_metadata["type"] != "cat")
        ],
        categorical_features=x_metadata["feature"][
            (x_metadata["type"] == "cat")
        ],
        fit_cache=cache,
        drift_logger=logger,
    )
    return detector


def run(
    drift_cache: DriftCache, dataset: Dataset, logger: DriftLogger
) -> None:
    print("Compute_drift.py")

    detector = get_drift_detector(dataset, drift_cache, logger)
    detector.fit_from_cache()

    x, _ = dataset.get_x_y()
    place_holder = np.zeros(len(x))
    detector.evaluate(
        x=x,
        t=place_holder,
        y=place_holder,
        y_scores=place_holder,
    )


def run_config(config: Dict[str, Any], current_time) -> None:
    dataset = get_dataset_from_config(config["test_dataset"])
    drift_cache = create_drift_cache(config["train_cache"])
    logger = get_logger(config["loggers"], current_time)
    run(drift_cache, dataset, logger)


if __name__ == "__main__":
    config_in = get_config()
    run_config(config_in, time.time())
    # days = pd.date_range("2016-01-01", "2017-01-03", freq="4W")
    # ts = days.values.astype(np.int64) // 10**9
    # for i, e in enumerate(ts):
    #     a = len(ts) // 10
    #     if (a > 0) and (i % a == 0):
    #         print(f"Running {i} out of {len(ts)}")
    #     run(config_in, e)
