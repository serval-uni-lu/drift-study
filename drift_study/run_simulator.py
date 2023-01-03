import logging
import os
import warnings
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Tuple

import configutils
import numpy as np
import pandas as pd
from configutils.utils import merge_parameters
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric
from optuna.exceptions import ExperimentalWarning
from tqdm import tqdm

from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_from_conf,
)
from drift_study.utils.delays import get_delays
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.helpers import (
    add_model,
    compute_y_scores,
    get_common_detectors_params,
    get_current_models,
    initialize,
)
from drift_study.utils.io_utils import save_drift_run

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", category=ExperimentalWarning, module="optuna.*"
)


def find_index_in_past(
    current_index: int, series: pd.Series, delta: Dict[str, int], past: bool
) -> int:
    current_date = series[current_index]
    if past:
        past_date = current_date - pd.DateOffset(**delta)
        return np.argwhere(series < past_date)[0][-1] + 1
    else:
        future_date = current_date + pd.DateOffset(**delta)
        return np.argwhere(series > future_date)[0][0] - 1


def run(
    config,
    run_i,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
    verbose=1,
) -> Tuple[int, float]:

    # CONFIG
    run_config = merge_parameters(
        config.get("common_runs_params"), config.get("runs")[run_i]
    )
    logger.info(f"Running config {run_config.get('name')}")

    # INITIALIZE PARAMETERS
    window_size = config.get("window_size")
    predict_forward = config.get("performance").get("predict_forward")

    # LOAD AND CREATE OBJECTS
    dataset, model, x, y, t = initialize(config, run_config)
    drift_detector = get_drift_detector_from_conf(
        run_config.get("detectors"),
        get_common_detectors_params(config, dataset.get_metadata(only_x=True)),
    )
    delays = get_delays(run_config, drift_detector)

    # PREPARE DATASTRUCTURES
    models: List[DriftModel] = []
    model_used = np.full(len(x), -1)
    if config["evaluation_params"]["n_score"] == 1:
        y_scores = np.full((len(x)), np.nan)
    else:
        y_scores = np.full(
            (len(x), config["evaluation_params"]["n_score"]), np.nan
        )
    is_drifts = np.full(len(x), np.nan)
    is_drift_warnings = np.full(len(x), np.nan)
    last_ml_model_used = 0
    last_drift_model_used = 0
    metrics = []

    # TRAIN FIRST MODEL
    model_path = (
        f"./models/{dataset.name}/{model.name}_{0}_{window_size}.joblib"
    )
    start_idx, end_idx = 0, window_size
    add_model(
        models,
        model_path,
        model,
        drift_detector,
        0,
        delays,
        x,
        y,
        t,
        start_idx,
        end_idx,
        lock_model_writing,
        list_model_writing,
    )

    # Main loop
    for x_idx in tqdm(np.arange(window_size, len(x)), disable=(verbose == 0)):

        # Find current model
        ml_model_idx, drift_model_idx = get_current_models(
            models, t[x_idx], last_ml_model_used, last_drift_model_used
        )
        last_ml_model_used, drift_model_idx = ml_model_idx, drift_model_idx
        # logger.debug(ml_model_idx)
        model = models[ml_model_idx].ml_model
        drift_detector = models[drift_model_idx].drift_detector

        # Update predictions if needed
        y_scores, model_used = compute_y_scores(
            model,
            x_idx,
            ml_model_idx,
            model_used,
            y_scores,
            x,
            predict_forward,
        )

        # Detect drift
        (
            is_drifts[x_idx],
            is_drift_warnings[x_idx],
            metric,
        ) = drift_detector.update(
            x=x.iloc[x_idx : x_idx + 1],
            t=t[x_idx : x_idx + 1],
            y=y[x_idx : x_idx + 1],
            y_scores=y_scores[x_idx : x_idx + 1],
        )
        metrics.append(metric)
        # Do not retrain if we are not using the latest drift model available
        # Due to delay
        if drift_model_idx == len(models) - 1:
            if is_drifts[x_idx]:
                logger.debug(f"Drift at index {x_idx}")
                start_idx = (x_idx + 1) - window_size
                end_idx = x_idx + 1
                logger.debug(f"start_index {start_idx}, end_index {end_idx}.")

                model_path = (
                    f"./models/{dataset.name}/"
                    f"{model.name}_{start_idx}_{end_idx}.joblib"
                )
                add_model(
                    models,
                    model_path,
                    model,
                    drift_detector,
                    x_idx,
                    delays,
                    x,
                    y,
                    t,
                    start_idx,
                    end_idx,
                    lock_model_writing,
                    list_model_writing,
                )

    # Save
    save_drift_run(
        numpy_to_save={
            "is_drifts": is_drifts,
            "is_drift_warnings": is_drift_warnings,
            "y_scores": y_scores,
            "model_used": model_used,
        },
        models=models,
        metrics=metrics,
        dataset_name=dataset.name,
        model_name=model.name,
        run_name=run_config.get("name"),
    )

    # Metrics
    n_train = int(np.max(model_used) + 1)
    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    if isinstance(prediction_metric, PredClassificationMetric):
        y_scores = np.argmax(y_scores, axis=1)
    metric = float(
        prediction_metric.compute(y[window_size:], y_scores[window_size:])
    )

    return n_train, metric


def run_many(
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    config_all = configutils.get_config()

    for i in range(len(config_all.get("runs"))):
        run(config_all, i, lock_model_writing, list_model_writing)


if __name__ == "__main__":
    run_many()
