import logging
import os
from pathlib import Path
from typing import List, Tuple

import configutils
import numpy as np
import pandas as pd
from mlc.models.model import Model
from numpy.typing import ArrayLike
from tqdm import tqdm

from drift_study.utils.drift_detector_factory import (
    get_drift_detector_from_conf,
)
from drift_study.utils.helpers import (
    compute_y_scores,
    get_current_models,
    get_model_arch,
    initialize,
    save_drift,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def load_do_save_model(model: Model, path: str, x, y) -> Model:
    if Path(path).exists():
        model.load(path)
        logger.info(f"Model {path} loaded.")
    else:
        logger.info(f"Fitting model {path}.")
        model.fit(x, y)
        model.save(path)
    return model


def run(config, run_i):

    run_config = config.get("runs")[run_i]

    dataset, model, x, y, t = initialize(config, run_config)
    window_size = config.get("window_size")
    metadata = dataset.get_metadata(only_x=True)
    auto_detector_params = {
        "features": metadata["feature"].to_list(),
        "numerical_features": metadata["feature"][
            metadata["type"] != "cat"
        ].to_list(),
        "categorical_features": metadata["feature"][
            metadata["type"] == "cat"
        ].to_list(),
        "window_size": window_size,
    }
    drift_detector = get_drift_detector_from_conf(
        run_config.get("detectors"),
        {**config.get("common_detectors_params"), **auto_detector_params},
    )

    # VARIABLES
    label_delay = pd.Timedelta(run_config.get("label_delay"))
    drift_detection_delay = pd.Timedelta(run_config.get("drift_delay"))
    if drift_detector.needs_label():
        drift_detection_delay = drift_detection_delay + label_delay
    retraining_delay = max(label_delay, drift_detection_delay) + pd.Timedelta(
        run_config.get("retraining_delay")
    )

    # DATA STRUCTURE
    models: List[Tuple[ArrayLike, Model, any, int, int]] = []

    # Train first model
    model_path = (
        f"./models/{dataset.name}/{model.name}_{0}_{window_size}.joblib"
    )
    start_index, end_index = 0, window_size
    model = load_do_save_model(
        model,
        model_path,
        x.iloc[start_index:end_index],
        y[start_index:end_index],
    )

    drift_detector.fit(
        x=x.iloc[:window_size],
        t=t[:window_size],
        y=y[:window_size],
        y_scores=model.predict(x.iloc[:window_size]),
        model=model,
    )

    models.append(
        (t[:window_size].iloc[-1], model, drift_detector, 0, window_size)
    )

    model_used = np.full(len(x), -1)
    y_scores = np.full((len(x), 2), np.nan)

    is_drifts = np.full(len(x), np.nan)
    is_drift_warnings = np.full(len(x), np.nan)

    predict_forward = config.get("performance").get("predict_forward")

    last_model_used = 0

    metrics = []

    for current_index in tqdm(np.arange(window_size, len(x))):

        # Find current model
        current_model_i = get_current_models(
            models, t[current_index], last_model_used
        )
        last_model_used = current_model_i
        model = models[current_model_i][1]
        drift_detector = models[current_model_i][2]

        # Update predictions if needed
        y_scores, model_used = compute_y_scores(
            model,
            current_index,
            current_model_i,
            model_used,
            y_scores,
            x,
            predict_forward,
        )

        # Detect drift
        (
            is_drifts[current_index],
            is_drift_warnings[current_index],
            metric,
        ) = drift_detector.update(
            x=x.iloc[current_index],
            t=t[current_index],
            y=y[current_index],
            y_scores=y_scores[current_index],
        )
        metrics.append(metric)
        # If we are already using the last model, retrain else not
        if current_model_i == len(models) - 1:
            if is_drifts[current_index]:
                logger.debug(f"Drift at index {current_index}")
                start_index = (current_index + 1) - window_size
                end_index = current_index + 1
                logger.debug(
                    f"start_index {start_index}, end_index {end_index}."
                )

                model_path = (
                    f"./models/{dataset.name}/"
                    f"{model.name}_{start_index}_{end_index}.joblib"
                )
                model = load_do_save_model(
                    get_model_arch(config, run_config, metadata),
                    model_path,
                    x=x.iloc[start_index:end_index],
                    y=y[start_index:end_index],
                )

                drift_detector.fit(
                    x=x.iloc[start_index:end_index],
                    t=t[start_index:end_index],
                    y=y[start_index:end_index],
                    y_scores=model.predict(x.iloc[start_index:end_index]),
                    model=model,
                )
                models.append(
                    (
                        t[current_index] + retraining_delay,
                        model,
                        drift_detector,
                        start_index,
                        end_index,
                    )
                )

    model_start_indexes = np.array(list(zip(*models))[3])
    model_end_indexes = np.array(list(zip(*models))[4])
    numpy_to_save = {
        "is_drifts": is_drifts,
        "is_drift_warnings": is_drift_warnings,
        "y_scores": y_scores,
        "model_used": model_used,
        "model_start_indexes": model_start_indexes,
        "model_end_indexes": model_end_indexes,
    }
    any_df = any([isinstance(e, pd.DataFrame) for e in metrics])
    if any_df:
        metrics = pd.concat(metrics).reset_index(drop=True)
    else:
        metrics = pd.DataFrame()

    save_drift(
        numpy_to_save,
        metrics,
        dataset.name,
        model.name,
        run_config.get("name"),
    )


def run_many():
    config_all = configutils.get_config()

    for i in range(len(config_all.get("runs"))):
        run(config_all, i)


if __name__ == "__main__":
    run_many()
