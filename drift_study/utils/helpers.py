import logging
import os
from pathlib import Path

import h5py
import joblib
import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset
from mlc.models.model import Model
from mlc.models.samples import load_model
from numpy.typing import ArrayLike
from sklearn.base import clone as sk_clone
from sklearn.pipeline import Pipeline

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def initialize(
    config,
    run_config,
) -> (Dataset, Model, ArrayLike, ArrayLike, ArrayLike):
    logger.info(f"Starting dataset {config.get('dataset').get('name')}")
    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()

    model = get_model_arch(
        config, run_config, dataset.get_metadata(only_x=True)
    )

    return dataset, model, x, y, t


def get_model_arch(config, run_config, metadata):
    model_name = run_config.get("model").get("name")

    model_class = load_model(model_name)
    model = model_class(
        name=model_name,
        x_metadata=metadata,
        verbose=0,
        n_jobs=config.get("performance").get("n_jobs"),
        random_state=config.get("experience").get("random_state"),
    )

    return model


def get_current_models(models, t, last_model_used=None):
    to_add = 0
    if last_model_used is not None:
        models = models[last_model_used:]
        to_add = last_model_used
    past_models = list(filter(lambda x: x[1][0] <= t, enumerate(models)))
    return past_models[-1][0] + to_add


def compute_y_scores(
    model,
    current_index,
    current_model_i,
    model_used,
    y_scores,
    x,
    predict_forward,
):
    if model_used[current_index] < current_model_i:
        logger.debug(f"Seeing forward at index {current_index}")
        y_scores[
            current_index : current_index + predict_forward
        ] = model.predict(x[current_index : current_index + predict_forward])
        model_used[
            current_index : current_index + predict_forward
        ] = current_model_i
    return y_scores, model_used


def clone_estimator(estimator) -> Pipeline:
    return sk_clone(estimator)


def save_models(models, model_path):
    for i, model in enumerate(list(zip(*models))[1]):
        joblib.dump(model, f"{model_path}_{i}.joblib")


def save_arrays(numpy_to_save, drift_data_path):
    Path(drift_data_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(drift_data_path, "w") as f:
        for key in numpy_to_save.keys():
            f.create_dataset(key, data=numpy_to_save[key], compression="gzip")


def save_drift(
    numpy_to_save, metrics: pd.DataFrame, dataset_name, model_name, run_name
):
    drift_data_path = f"./data/{dataset_name}/drift/{model_name}_{run_name}"

    save_arrays(numpy_to_save, drift_data_path)
    metrics.to_hdf(f"{drift_data_path}_metrics.hdf5", "metrics")

    # save_models(
    #     models,
    #     f"./models/{dataset_name}/{model_name}_{run_name}",
    # )


def get_ref_eval_config(configs, ref_config_names):
    ref_configs = []
    eval_configs = []
    for config in configs.get("runs"):
        if config.get("name") in ref_config_names:
            ref_configs.append(config)
        else:
            eval_configs.append(config)
    return ref_configs, eval_configs
