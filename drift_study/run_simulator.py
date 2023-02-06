import logging
import os
import warnings
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import configutils
import numpy as np
from configutils.utils import merge_parameters
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric
from optuna.exceptions import ExperimentalWarning
from tqdm import tqdm

from drift_study.utils.delays import get_delays
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.helpers import (
    add_model,
    compute_y_scores,
    free_mem_models,
    get_current_models,
    initialize,
)
from drift_study.utils.io_utils import save_drift_run

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore", category=ExperimentalWarning, module="optuna.*"
)


def run(
    config,
    run_i,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
    verbose=1,
) -> Tuple[int, Union[float, List[float]]]:
    # CONFIG
    run_config = merge_parameters(
        config.get("common_runs_params"), config.get("runs")[run_i]
    )
    logger.info(f"Running config {run_config.get('name')}")
    model_root_dir = config.get(
        "models_dir", os.environ.get("MODELS_DIR", "./models")
    )

    # INITIALIZE PARAMETERS
    window_size = config.get("window_size")
    predict_forward = config.get("performance").get("predict_forward")
    n_early_stopping = config.get("evaluation_params", {}).get(
        "n_early_stopping", 0
    )
    early_stopped = False

    # LOAD AND CREATE OBJECTS
    dataset, f_new_model, f_new_detector, x, y, t = initialize(
        config, run_config
    )

    delays = get_delays(run_config, f_new_detector())
    last_idx = config["evaluation_params"].get("last_idx", -1)
    if last_idx == -1:
        last_idx = len(x)

    # PREPARE DATASTRUCTURES
    models: List[DriftModel] = []
    model_used = np.full(last_idx, -1)
    if config["evaluation_params"]["n_score"] == 1:
        y_scores = np.full(last_idx, np.nan)
    else:
        y_scores = np.full(
            (last_idx, config["evaluation_params"]["n_score"]), np.nan
        )
    is_drifts = np.full(last_idx, np.nan)
    is_drift_warnings = np.full(last_idx, np.nan)
    last_ml_model_used = 0
    last_drift_model_used = 0
    metrics = []
    model_name = f_new_model().name

    # TRAIN FIRST MODEL
    model_path = (
        f"{model_root_dir}/{dataset.name}/"
        f"{model_name}_{0}_{window_size}.joblib"
    )
    start_idx, end_idx = 0, window_size
    add_model(
        models,
        model_path,
        f_new_model,
        f_new_detector,
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
    for x_idx in tqdm(
        np.arange(window_size, last_idx), disable=(verbose == 0)
    ):

        # Find current model
        ml_model_idx, drift_model_idx = get_current_models(
            models, t[x_idx], last_ml_model_used, last_drift_model_used
        )
        last_ml_model_used, last_drift_model_used = (
            ml_model_idx,
            drift_model_idx,
        )
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
            last_idx,
        )
        if not early_stopped:
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
            # Do not retrain if we are
            # not using the latest drift model available
            # Due to delay
            if drift_model_idx == len(models) - 1:
                if is_drifts[x_idx]:
                    logger.debug(f"Drift at index {x_idx}")
                    start_idx = (x_idx + 1) - window_size
                    end_idx = x_idx + 1
                    logger.debug(
                        f"start_index {start_idx}, end_index {end_idx}."
                    )

                    model_path = (
                        f"{model_root_dir}/{dataset.name}/"
                        f"{model_name}_{start_idx}_{end_idx}.joblib"
                    )
                    add_model(
                        models,
                        model_path,
                        f_new_model,
                        f_new_detector,
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
                    if len(models) > 1:
                        assert (
                            models[-1].drift_detector
                            != models[-2].drift_detector
                        )
                        assert models[-1].ml_model != models[-2].ml_model

                    # Avoid memory full
                    free_mem_models(models, ml_model_idx, drift_model_idx)
                    # Early stop
                    if (0 <= n_early_stopping) and (
                        n_early_stopping <= len(models)
                    ):
                        early_stopped = True

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
        model_name=model_name,
        run_name=run_config.get("name"),
        sub_dir_path=config["sub_dir_path"],
    )

    # Metrics
    n_train = int(np.max(model_used) + 1)
    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    if isinstance(prediction_metric, PredClassificationMetric):
        y_scores = np.argmax(y_scores, axis=1)

    val_test_idx = config["evaluation_params"].get("val_test_idx")
    if val_test_idx is None:
        metric = float(
            prediction_metric.compute(
                y[window_size:last_idx], y_scores[window_size:last_idx]
            )
        )
    else:
        metric_idxs = [
            (window_size, last_idx),
            (window_size, val_test_idx),
            (val_test_idx, last_idx),
        ]
        metric = [
            float(
                prediction_metric.compute(
                    y[m_idx_start:m_idx_end], y_scores[m_idx_start:m_idx_end]
                )
            )
            for m_idx_start, m_idx_end in metric_idxs
        ]

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
