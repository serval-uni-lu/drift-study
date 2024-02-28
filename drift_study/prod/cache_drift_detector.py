from typing import Any, Dict, Optional

import configutils
import joblib
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset_from_config
from sklearn.ensemble import RandomForestClassifier

from drift_study.drift_cache.factory import create_drift_cache
from drift_study.drift_detectors.data_based.divergence_drift import (
    DivergenceDrift,
)
from drift_study.drift_detectors.predictive.rf_bayesian_uncertainty.rf_uncertainty_drift_prod import (
    RfUncertaintyDrift,
)


def run(dataset: Dataset, model_path: str, n_train: Optional[int] = None):
    x, y, t = dataset.get_x_y_t()
    if n_train is None:
        n_train = len(x)

    x_train = x.iloc[:n_train]
    y_train = y[:n_train]
    x_metadata = dataset.get_metadata(only_x=True)

    cache = create_drift_cache(config["train_cache"]["divergence"])
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

    # Uncertainty drift detector

    random_forest = RandomForestClassifier(
        n_estimators=10, class_weight="balanced", max_depth=5, random_state=42
    )

    random_forest.fit(x_train, y_train)
    joblib.dump(random_forest, model_path)

    cache = create_drift_cache(config["train_cache"]["uncertainty"])
    detector = RfUncertaintyDrift(None, x_metadata, "total", fit_cache=cache)
    detector.fit(
        x_train, t[:n_train], y[:n_train], y[:n_train], model=random_forest
    )
    detector.save_cache()


def run_config(config: Dict[str, Any]) -> None:
    print("Start cache.")
    dataset = get_dataset_from_config(config["train_dataset"])
    model_path = config["model_path"]
    n_train = config.get("n_train")

    run(
        dataset,
        model_path,
        n_train,
    )


if __name__ == "__main__":
    config = configutils.get_config()
    run_config(config)
