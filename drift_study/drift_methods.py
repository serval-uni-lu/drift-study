import logging
import os

import h5py
import joblib
import numpy as np
from mlc.datasets import load_datasets
from mlc.models import load_models
from sklearn.base import clone as clone_estimator
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from drift_study.drift_detectors import load_drift_detectors

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


# Place here parameters that do not influence results
performance_params = {
    "predict_forward": 500000,
    "n_jobs": 10,
}

common_params = {
    "window_size": 100000,
    "batch_size": 5000,
    "random_state": 42,
    "p_value": 0.05,
    "period": 5000,
}

configs = [
    # {
    #     "dataset_name": "lcld",
    #     "model_name": "random_forest",
    #     "drift_name": "periodic",
    # },
    # {
    #     "dataset_name": "lcld",
    #     "model_name": "random_forest",
    #     "drift_name": "adwin",
    # },
    # {
    #     "dataset_name": "lcld",
    #     "model_name": "random_forest",
    #     "drift_name": "evidently",
    # },
    {
        "dataset_name": "lcld",
        "model_name": "random_forest",
        "drift_name": "no_drift",
    },
]


def run(config):
    print(config)
    dataset_name = config.get("dataset_name")
    model_name = config.get("model_name")
    logger.info(f"Starting dataset {dataset_name}, model {model_name}")
    dataset = load_datasets(dataset_name)
    x, y, t = dataset.get_x_y_t()

    model = load_models(dataset_name, model_name)
    model.set_params(
        **{
            "verbose": 1,
            "n_jobs": performance_params.get("n_jobs", 1),
            "random_state": common_params.get("random_state"),
        }
    )
    pipeline = Pipeline(
        steps=[("pre_process", dataset.get_preprocessor()), ("model", model)]
    )
    drift_detector = load_drift_detectors(config.get("drift_name"))
    window_size = common_params.get("window_size")
    drift_detector = drift_detector(
        window_size=window_size,
        batch=common_params.get("batch_size"),
        p_val=common_params.get("p_value"),
        features=x.columns,
        numerical_features=dataset.numerical_features,
        categorical_features=dataset.categorical_features,
        period=common_params.get("period"),
        batch_size=common_params.get("batch_size"),
    )

    fitted_models = [pipeline.fit(x.iloc[:window_size], y[:window_size])]
    pipeline[-1].set_params(**{"verbose": 0})
    drift_detector.fit(
        x.iloc[:window_size],
        y[:window_size],
        fitted_models[0].predict_proba(x.iloc[:window_size]),
    )

    model_used = np.full(len(x), -1)
    y_scores = np.full((len(x), 2), np.nan)

    current_model = 0

    is_drifts = np.full(len(x), np.nan)
    is_drift_warnings = np.full(len(x), np.nan)
    drift_distances = np.full(len(x), np.nan)
    drift_p_values = np.full(len(x), np.nan)

    predict_forward = performance_params.get("predict_forward")

    for current_index in tqdm(np.arange(window_size, len(x))):
        if model_used[current_index] < current_model:
            logger.debug(f"Seeing forward at index {current_index}")
            y_scores[
                current_index : current_index + predict_forward
            ] = fitted_models[current_model].predict_proba(
                x[current_index : current_index + predict_forward]
            )
            model_used[
                current_index : current_index + predict_forward
            ] = current_model

        (
            is_drifts[current_index],
            is_drift_warnings[current_index],
            drift_distances[current_index],
            drift_p_values[current_index],
        ) = drift_detector.update(
            x.iloc[current_index], y[current_index], y_scores[current_index]
        )

        if is_drifts[current_index]:
            logger.debug(f"Drift at index {current_index}")
            start_index = (current_index + 1) - window_size
            end_index = current_index + 1
            fitted_models.append(
                clone_estimator(pipeline).fit(
                    x.iloc[start_index:end_index], y[start_index:end_index]
                )
            )
            current_model += 1
            drift_detector.fit(
                x.iloc[start_index:end_index],
                y[start_index:end_index],
                fitted_models[current_model].predict_proba(
                    x.iloc[start_index:end_index]
                ),
            )

    drift_data_path = (
        f"./data/{dataset_name}/drift/{model_name}_{config.get('drift_name')}"
    )
    with h5py.File(drift_data_path, "w") as f:
        f.create_dataset("is_drifts", data=is_drifts, compression="gzip")
        f.create_dataset(
            "is_drift_warnings", data=is_drift_warnings, compression="gzip"
        )
        f.create_dataset(
            "drift_distances", data=drift_distances, compression="gzip"
        )
        f.create_dataset(
            "drift_p_values", data=drift_p_values, compression="gzip"
        )
        f.create_dataset("y_scores", data=y_scores, compression="gzip")
        f.create_dataset("model_used", data=model_used, compression="gzip")

    model_path = (
        f"./models/{dataset_name}/{model_name}_{config.get('drift_name')}"
    )
    for i, fitted_model in enumerate(fitted_models):
        joblib.dump(fitted_model, f"{model_path}_{i}.joblib")


def run_many(configs_l):
    for config in configs_l:
        run(config)


if __name__ == "__main__":
    run_many(configs)
