import logging
import time
from multiprocessing import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlc.load_do_save import save_json
from mlc.models.model import Model

from drift_study.utils.run_results import RunResult


def attempt_read(path: str, model: Model) -> int:
    logger = logging.getLogger(__name__)

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
    logger = logging.getLogger(__name__)

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
    drift_data_path: str,
    config: Dict[str, Any],
    run_result: RunResult,
) -> None:

    Path(drift_data_path).mkdir(parents=True, exist_ok=True)
    # Save config
    save_json(config, f"{drift_data_path}/config.json")

    # Save preds
    schedule = pd.DataFrame(
        np.array(
            [
                run_result.is_drifts,
                run_result.is_drift_warnings,
                run_result.model_used,
            ]
        ).T,
        columns=["is_drifts", "is_drift_warnings", "model_used"],
    )
    schedule.to_parquet(f"{drift_data_path}/schedule.parquet")

    preds = pd.DataFrame(
        run_result.y_scores,
        columns=[str(e) for e in range(run_result.y_scores.shape[1])],
    )
    preds.to_parquet(f"{drift_data_path}/preds.parquet")

    models_idxs = pd.DataFrame(
        np.array([run_result.model_start_idxs, run_result.model_end_idxs]).T,
        columns=["model_start_idxs", "model_end_idxs"],
    )
    models_idxs.to_parquet(f"{drift_data_path}/models_idxs.parquet")

    # any_df = any([isinstance(e, pd.DataFrame) for e in metrics])
    # if any_df:
    #     metrics = pd.concat(metrics).reset_index(drop=True)
    # else:
    #     metrics = pd.DataFrame()

    # metrics.to_hdf(f"{drift_data_path}_metrics.hdf5", "metrics")

    save_json(
        {"n_train": run_result.n_train, "ml_metric": run_result.ml_metric},
        f"{drift_data_path}/metrics.json",
    )


def check_parent_path(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def manual_save_run(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    n_train: Union[int, List[int]],
    ml_metric: Union[float, List[float]],
) -> None:
    out = {"config": config, "n_train": n_train, "ml_metric": ml_metric}

    model_name = run_config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    sub_dir_path = config["sub_dir_path"]

    out_path = (
        f"./data/optimizer_results/"
        f"{dataset_name}/{model_name}/"
        f"{sub_dir_path}/{run_config['name']}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)
