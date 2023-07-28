from typing import Any, Callable, Dict

import pandas as pd

from drift_study.drift_detectors.drift_detector import DriftDetector
from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_from_conf,
)


def get_auto_detectors_params(
    metadata: pd.DataFrame, start_idx: int, end_idx: int
):
    auto_detector_params = {
        "x_metadata": metadata,
        "features": metadata["feature"].to_list(),
        "numerical_features": metadata["feature"][
            metadata["type"] != "cat"
        ].to_list(),
        "categorical_features": metadata["feature"][
            metadata["type"] == "cat"
        ].to_list(),
        "start_idx": start_idx,
        "end_idx": end_idx,
    }
    return auto_detector_params


def get_f_new_detector(
    config: Dict[str, Any],
    metadata_x: pd.DataFrame,
) -> Callable[[int, int], DriftDetector]:
    def f_new_detector(start_idx, end_idx) -> DriftDetector:
        drift_detector = get_drift_detector_from_conf(
            config,
            get_auto_detectors_params(metadata_x, start_idx, end_idx),
        )
        return drift_detector

    return f_new_detector
