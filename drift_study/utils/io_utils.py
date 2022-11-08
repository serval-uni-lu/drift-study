import logging
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from mlc.models.model import Model

logger = logging.getLogger(__name__)


def load_do_save_model(model: Model, path: str, x, y) -> Model:
    if Path(path).exists():
        model.load(path)
        logger.debug(f"Model {path} loaded.")
    else:
        logger.debug(f"Fitting model {path}.")
        model.fit(x, y)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
    return model


def save_models(models, model_path):
    for i, model in enumerate(list(zip(*models))[1]):
        joblib.dump(model, f"{model_path}_{i}.joblib")


def save_arrays(numpy_to_save, drift_data_path):
    Path(drift_data_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(drift_data_path, "w") as f:
        for key in numpy_to_save.keys():
            f.create_dataset(key, data=numpy_to_save[key], compression="gzip")


def save_drift_run(
    numpy_to_save, models, metrics, dataset_name, model_name, run_name
):
    model_start_indexes = np.array([model.start_idx for model in models])
    model_end_indexes = np.array([model.end_idx for model in models])
    numpy_to_save["model_start_indexes"] = model_start_indexes
    numpy_to_save["model_end_indexes"] = model_end_indexes

    any_df = any([isinstance(e, pd.DataFrame) for e in metrics])
    if any_df:
        metrics = pd.concat(metrics).reset_index(drop=True)
    else:
        metrics = pd.DataFrame()

    drift_data_path = (
        f"./data/drift-study/{dataset_name}/{model_name}/{run_name}.hdf5"
    )

    save_arrays(numpy_to_save, drift_data_path)
    metrics.to_hdf(f"{drift_data_path}_metrics.hdf5", "metrics")
