import logging
import os

import configutils
import numpy as np
import pandas as pd
from mlc.load_do_save import load_do_save
from sklearn.base import BaseEstimator
from tqdm import tqdm

from drift_study.utils.drift_detector_factory import get_drift_detector
from drift_study.utils.helpers import (
    clone_estimator,
    compute_y_scores,
    get_current_models,
    initialize,
    save_drift,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run(config, common_params, performance_params):
    dataset, model, x, y, t = initialize(
        config, common_params, performance_params
    )
    dataset_name = config.get("dataset_name")
    model_name = config.get("model_name")

    window_size = common_params.get("window_size")
    drift_detector = get_drift_detector(
        drift_detectors_names=config.get("drift_name"),
        window_size=window_size,
        batch=common_params.get("batch_size"),
        p_val=common_params.get("p_value"),
        features=x.columns,
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        period=common_params.get("period"),
        batch_size=common_params.get("batch_size"),
        **config.get("drift_parameters", {}),
    )

    # VARIABLES
    label_delay = pd.Timedelta(config.get("label_delay"))
    drift_detection_delay = pd.Timedelta(config.get("drift_delay"))
    if drift_detector.needs_label():
        drift_detection_delay = drift_detection_delay + label_delay
    retraining_delay = max(label_delay, drift_detection_delay) + pd.Timedelta(
        config.get("retraining_delay")
    )

    # DATA STRUCTURE
    models = []

    # Train first model

    model_path = (
        f"./models/{dataset_name}/{model_name}_{0}_{window_size}.joblib"
    )
    start_index, end_index = 0, window_size

    def fit_l() -> BaseEstimator:
        model.fit(x.iloc[start_index:end_index], y[start_index:end_index])
        return model

    model = load_do_save(path=model_path, executable=fit_l, verbose=True)
    drift_detector.fit(
        x=x.iloc[:window_size],
        t=t[:window_size],
        y=y[:window_size],
        y_scores=model.predict_proba(x.iloc[:window_size]),
        model=model,
    )

    models.append(
        (t[:window_size].iloc[-1], model, drift_detector, 0, window_size)
    )

    model_used = np.full(len(x), -1)
    y_scores = np.full((len(x), 2), np.nan)

    is_drifts = np.full(len(x), np.nan)
    is_drift_warnings = np.full(len(x), np.nan)

    predict_forward = performance_params.get("predict_forward")

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

                model = clone_estimator(model)

                model_path = (
                    f"./models/{dataset_name}/"
                    f"{model_name}_{start_index}_{end_index}.joblib"
                )

                def fit_l() -> BaseEstimator:
                    model.fit(
                        x.iloc[start_index:end_index], y[start_index:end_index]
                    )
                    return model

                model = load_do_save(
                    path=model_path, executable=fit_l, verbose=True
                )

                drift_detector.fit(
                    x=x.iloc[start_index:end_index],
                    t=t[start_index:end_index],
                    y=y[start_index:end_index],
                    y_scores=model.predict_proba(
                        x.iloc[start_index:end_index]
                    ),
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
        dataset_name,
        model_name,
        config.get("run_name"),
    )


def run_many(configs_l, common_params, performance_params):
    for config in configs_l:
        run(config, common_params, performance_params)


if __name__ == "__main__":
    configs = configutils.get_config()
    runs = configs.get("runs")
    common_params_l = configs.get("common_params")
    performance_params_l = configs.get("performance_params")
    run_many(runs, common_params_l, performance_params_l)
