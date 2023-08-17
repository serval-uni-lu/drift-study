import logging
import os
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Tuple

import configutils
import joblib
import numpy as np
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from sklearn.model_selection import TimeSeriesSplit

from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.run.no_retrain_baseline import add_best_params_to_model
from drift_study.run_simulator import get_start_end_idx
from drift_study.utils.helpers import get_f_new_model
from drift_study.utils.io_utils import load_do_save_model
from drift_study.utils.logging import configure_logger


def train_model(
    f_new_model,
    model_path,
    x,
    y,
    start_idx,
    end_idx,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
):
    model = f_new_model()
    x = x.to_numpy()
    model = load_do_save_model(
        model,
        model_path,
        x[start_idx:end_idx],
        y[start_idx:end_idx],
        lock_model_writing,
        list_model_writing,
    )
    if isinstance(model, LazyPipeline):
        model.safe_lazy_predict(x, start_idx, end_idx)
        model.safe_lazy_many_predict(
            x,
            50,
            start_idx,
            end_idx,
        )


def get_idx_window_size_period(
    windows_sizes: List[int],
    periods: List[int],
    test_start_idx: int,
    last_idx: int,
) -> List[Tuple[int, int]]:
    train_idxs: List[Tuple[int, int]] = []
    if isinstance(windows_sizes, int):
        windows_sizes = [windows_sizes]
    for window_size in windows_sizes:
        for period in periods:
            for i in range(test_start_idx, last_idx + 1, period):
                train_idxs.append(get_start_end_idx(i, window_size))
    return train_idxs


def get_pretrains(
    config: Dict[str, Any], x: pd.DataFrame
) -> List[Tuple[int, int]]:
    train_idxs: List[Tuple[int, int]] = []
    test_start_idx = int(config["test_start_idx"])
    last_idx = config.get("last_idx", len(x))
    if config.get("pretrain_periodic"):
        train_idxs.extend(
            get_idx_window_size_period(
                config.get("train_window_sizes"),
                config.get("periods"),
                test_start_idx,
                last_idx,
            )
        )

    if config.get("pretrain_schedules"):
        train_idxs.extend(
            get_idx_window_size_period(
                config.get("train_window_size"),
                [config.get("training_step_size")],
                test_start_idx,
                last_idx,
            )
        )
    if config.get("pretrain_schedules_opt"):
        tss = TimeSeriesSplit(
            n_splits=config["optimization_splits"]["detector"]
        )
        for i, (train_idx, test_idx) in enumerate(
            tss.split(np.arange(test_start_idx))
        ):
            local_test_start_idx = test_idx[0]
            local_last_idx = test_start_idx
            train_idxs.append((0, local_test_start_idx))
            train_idxs.extend(
                get_idx_window_size_period(
                    config.get("train_window_size"),
                    [config.get("training_step_size")],
                    local_test_start_idx,
                    local_last_idx,
                )
            )
            print(train_idxs)

    return list(set(train_idxs))


def run(config: Dict[str, Any]) -> None:
    if config.get("use_auto_model_tuning"):
        config = add_best_params_to_model(config)
    print(config["model"]["params"])
    logger = logging.getLogger(__name__)
    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()
    metadata = dataset.get_metadata(only_x=True)
    f_new_model = get_f_new_model(config["model"], metadata)

    idx_to_train = get_pretrains(config, x)
    if config.get("dry_run"):
        logger.info("Dry run, not training any models")
        logger.info(f"Training {len(idx_to_train)} models")
        for start_idx, end_idx in idx_to_train:
            logger.info(f"Training model from {start_idx} to {end_idx}")
    else:
        logger.info("Training models")
        model_root_dir = config.get(
            "models_dir", os.environ.get("MODELS_DIR", "./models")
        )
        model_name = f_new_model().name

        def train_model_wrapper(
            i,
            f_new_model,
            x,
            y,
            start_idx,
            end_idx,
        ) -> None:
            logger.info(
                f"Training model {i}/ {len(idx_to_train)} "
                f"from {start_idx} to {end_idx}"
            )
            model_path = model_path = (
                f"{model_root_dir}/{dataset.name}/"
                f"{model_name}_{start_idx}_{end_idx}.joblib"
            )
            train_model(
                f_new_model,
                model_path,
                x,
                y,
                start_idx,
                end_idx,
            )

        if joblib.cpu_count() > 4:
            n_jobs = joblib.cpu_count() // 4
            # Parrallel training with joblib
            joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(train_model_wrapper)(
                    i,
                    f_new_model,
                    x,
                    y,
                    start_idx,
                    end_idx,
                )
                for i, (start_idx, end_idx) in enumerate(idx_to_train)
            )
        else:
            for i, (start_idx, end_idx) in enumerate(idx_to_train):
                train_model_wrapper(
                    i,
                    f_new_model,
                    x,
                    y,
                    start_idx,
                    end_idx,
                )


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)
    run(config)
