import logging
import os
from typing import List

import configutils
import numpy as np
from tqdm import tqdm

from drift_study.utils.drift_detector_factory import (
    get_drift_detector_from_conf,
)
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.helpers import (
    add_model,
    compute_y_scores,
    get_common_detectors_params,
    get_current_models,
    get_delays,
    initialize,
)
from drift_study.utils.io_utils import save_drift_run

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run(config, run_i):

    run_config = config.get("runs")[run_i]
    logger.info(f"Running config {run_config.get('name')}")
    dataset, model, x, y, t = initialize(config, run_config)
    window_size = config.get("window_size")
    metadata = dataset.get_metadata(only_x=True)
    drift_detector = get_drift_detector_from_conf(
        run_config.get("detectors"),
        get_common_detectors_params(config, metadata),
    )

    label_delay, drift_detection_delay, retraining_delay = get_delays(
        run_config, drift_detector
    )

    # DATA STRUCTURE
    models: List[DriftModel] = []

    # Train first model
    model_path = (
        f"./models/{dataset.name}/{model.name}_{0}_{window_size}.joblib"
    )
    start_idx, end_idx = 0, window_size
    t_available = t[:window_size].iloc[-1]
    add_model(
        models,
        model_path,
        model,
        drift_detector,
        t_available,
        x,
        y,
        t,
        start_idx,
        end_idx,
    )

    # Running var
    model_used = np.full(len(x), -1)
    if config["evaluation_params"]["n_score"] == 1:
        y_scores = np.full((len(x)), np.nan)
    else:
        y_scores = np.full(
            (len(x), config["evaluation_params"]["n_score"]), np.nan
        )
    is_drifts = np.full(len(x), np.nan)
    is_drift_warnings = np.full(len(x), np.nan)
    predict_forward = config.get("performance").get("predict_forward")
    last_model_used = 0
    metrics = []

    # Main loop
    for x_idx in tqdm(np.arange(window_size, len(x))):

        # Find current model
        model_idx = get_current_models(models, t[x_idx], last_model_used)
        last_model_used = model_idx
        model = models[model_idx].ml_model
        drift_detector = models[model_idx].drift_detector

        # Update predictions if needed
        y_scores, model_used = compute_y_scores(
            model,
            x_idx,
            model_idx,
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
            x=x.iloc[x_idx],
            t=t[x_idx],
            y=y[x_idx],
            y_scores=y_scores[x_idx],
        )
        metrics.append(metric)
        # Do not retrain if we are not using the latest model available
        # Due to delay
        if model_idx == len(models) - 1:
            if is_drifts[x_idx]:
                logger.debug(f"Drift at index {x_idx}")
                start_idx = (x_idx + 1) - window_size
                end_idx = x_idx + 1
                logger.debug(f"start_index {start_idx}, end_index {end_idx}.")

                model_path = (
                    f"./models/{dataset.name}/"
                    f"{model.name}_{start_idx}_{end_idx}.joblib"
                )
                t_available = t[x_idx] + retraining_delay
                add_model(
                    models,
                    model_path,
                    model,
                    drift_detector,
                    t_available,
                    x,
                    y,
                    t,
                    start_idx,
                    end_idx,
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


def run_many():
    config_all = configutils.get_config()

    for i in range(len(config_all.get("runs"))):
        run(config_all, i)


if __name__ == "__main__":
    run_many()
