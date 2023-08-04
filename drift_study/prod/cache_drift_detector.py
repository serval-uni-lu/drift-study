from typing import Any, Dict, Optional

import configutils
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset_from_config

from drift_study.drift_cache.drift_cache import DriftCache
from drift_study.drift_cache.factory import create_drift_cache
from drift_study.drift_detectors.data_based.divergence_drift import (
    DivergenceDrift,
)


def run(dataset: Dataset, cache: DriftCache, n_train: Optional[int] = None):
    x, y, t = dataset.get_x_y_t()
    if n_train is None:
        n_train = len(x)

    x_train = x.iloc[:n_train]
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
    )
    detector.fit(x_train, t[:n_train], y[:n_train], y[:n_train], None)
    detector.save_cache()


def run_config(config: Dict[str, Any]) -> None:
    print("Start cache.")
    dataset = get_dataset_from_config(config["train_dataset"])
    n_train = config.get("n_train")
    cache = create_drift_cache(config["train_cache"])

    run(dataset, cache, n_train)


if __name__ == "__main__":
    config = configutils.get_config()
    run_config(config)
