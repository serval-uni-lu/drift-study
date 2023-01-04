import logging
import time
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlc.models.model import Model

logger = logging.getLogger(__name__)


def attempt_read(path: str, model: Model) -> int:
    try:
        model.load(path)
        logger.debug(f"Model {path} loaded.")
        return 0
    except (ValueError, EOFError):
        logger.debug(f"Model {path} error while loading, attempt to fix.")
        return -1


def load_do_save_model(
    model: Model,
    path: str,
    x: Union[npt.NDArray, pd.DataFrame],
    y: Union[npt.NDArray, pd.Series],
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Model:

    if lock_model_writing is None:
        read = Path(path).exists()
        write = not read
        if read:
            write = attempt_read(path, model) != 0
        if write:
            logger.debug(f"Fitting model {path}.")
            model.fit(x, y)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
        return model
    else:
        write = False
        read = False
        while not (read or write):
            with lock_model_writing:
                if path not in list_model_writing:
                    if Path(path).exists():
                        # LOAD
                        read = True
                    else:
                        # SAVE
                        write = True
                    list_model_writing[path] = True
            if not (read or write):
                logger.debug(f"Model {path} waiting...")
                time.sleep(1)

        if read:
            if attempt_read(path, model) != 0:
                write = True
            else:
                list_model_writing.pop(path)
                return model

        if write:
            model.fit(x, y)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
            logger.debug(f"Model {path} fitted.")
            list_model_writing.pop(path)
            return model

        raise NotImplementedError


def save_models(models, model_path: str) -> None:
    for i, model in enumerate(list(zip(*models))[1]):
        joblib.dump(model, f"{model_path}_{i}.joblib")


def save_arrays(
    numpy_to_save: Dict[str, npt.NDArray[Any]], drift_data_path: str
) -> None:
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
