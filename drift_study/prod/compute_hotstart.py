import time
from typing import Any, Dict, List, Tuple

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


def get_logger(logger_config: List[Dict[str, Any]]) -> DriftLogger:
    loggers = [create_drift_logger(c) for c in logger_config]
    logger = MultiDriftLogger(loggers)
    return logger


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


def get_date_batches(start_date, end_date, date_round, test_window, freq):

    if date_round:
        start_date_round = start_date.floor(date_round)
        end_date_round = end_date.ceil(date_round)

    if isinstance(test_window, str):
        test_window = pd.Timedelta(test_window)

    out = []
    for i in pd.date_range(
        start_date_round,
        end_date_round,
        freq=freq,
    ):
        start = i
        end = i + test_window
        out.append((start, end))
        if end > end_date_round:
            break
    return out


def run(
    drift_cache: DriftCache,
    dataset: Dataset,
    logger: DriftLogger,
    date_batches: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    print("Compute_hotstart.py")

    detector = get_drift_detector(dataset, drift_cache, logger)
    detector.fit_from_cache()
    x, _, t = dataset.get_x_y_t()

    for i, (start, end) in enumerate(date_batches):
        detector.drift_logger = StaticTimeDriftLogger(
            logger, end.value / 10**9
        )
        print(f"Batch {i+1}/{len(date_batches)}: [{start}, {end}[")
        x_batch = x[t.between(start, end, inclusive="left")]
        place_holder = np.zeros(len(x_batch))
        detector.evaluate(
            x=x_batch,
            t=place_holder,
            y=place_holder,
            y_scores=place_holder,
        )


def run_config(config: Dict[str, Any]) -> None:
    dataset = get_dataset_from_config(config["hotstart_dataset"])
    drift_cache = create_drift_cache(config["train_cache"])
    logger = get_logger(config["loggers"])
    _, _, t = dataset.get_x_y_t()
    date_batches = get_date_batches(
        t.iloc[0],
        t.iloc[len(t) - 1],
        config.get("date_round"),
        config.get("testing_window"),
        config.get("hotstart_freq"),
    )
    run(drift_cache, dataset, logger, date_batches)


if __name__ == "__main__":
    config_in = get_config()
    run_config(config_in)
