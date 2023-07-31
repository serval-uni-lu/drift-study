import copy
import logging.config
import os
from multiprocessing import Manager
from multiprocessing.synchronize import Lock as LockType
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import configutils
import joblib
import numpy as np
import numpy.typing as npt
import optuna
from configutils.utils import merge_parameters
from joblib import Parallel, delayed, parallel_backend
from mlc.datasets.dataset_factory import get_dataset
from mlc.load_do_save import load_json, save_json
from optuna import Study
from optuna._callbacks import MaxTrialsCallback, RetryFailedTrialCallback
from optuna.samplers import TPESampler
from optuna.trial import FrozenTrial, TrialState
from sklearn.model_selection import TimeSeriesSplit

from drift_study import run_simulator
from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_class_from_conf,
)
from drift_study.run.no_retrain_baseline import add_best_params_to_model
from drift_study.typing import NDInt
from drift_study.utils.logging import configure_logger


def get_default_params(
    config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> Dict[str, Any]:
    out = {}
    for i, e in enumerate(list_drift_detector):
        out = merge_parameters(
            out,
            e["detector"].get_default_params(
                trial_params=config["trial_params"]
            ),
        )
    return out


def update_config_params(
    trial: optuna.Trial,
    config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> None:
    logger = logging.getLogger(__name__)

    config["schedule"][
        "name"
    ] = f"{config['schedule']['name']}_iter_{trial.number}"
    config[
        "schedule_data_path"
    ] = f"{config['schedule_data_path']}/schedules/iter_{trial.number}"
    detectors_config = config["schedule"]["detectors"]

    for i, e in enumerate(list_drift_detector):
        new_params = merge_parameters(
            e.get("params", {}),
            e["detector"].define_trial_parameters(
                trial, trial_params=config["trial_params"]
            ),
        )
        logger.debug(new_params)
        detectors_config[i]["params"] = new_params


def update_fold_params(
    config: Dict[str, Any], fold_idx: int, test_idxs: NDInt
) -> None:
    config["schedule"][
        "name"
    ] = f"{config['schedule']['name']}_fold_{fold_idx}"
    config[
        "schedule_data_path"
    ] = f"{config['schedule_data_path']}_fold_{fold_idx}"
    config["last_idx"] = config["test_start_idx"]
    config["test_start_idx"] = int(test_idxs[0])


def add_auto_trial_params(config: Dict[str, Any]) -> None:
    ds = get_dataset(config["dataset"])
    n_feature = len(ds.get_metadata(only_x=True))
    config["trial_params"]["n_features"] = n_feature


def execute_one_fold(
    fold_idx: int,
    config: Dict[str, Any],
    train_idxs: npt.NDArray[np.int_],
    test_idxs: npt.NDArray[np.int_],
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float]:

    logger = logging.getLogger(__name__)
    update_fold_params(config, fold_idx, test_idxs)
    logger.info(f"Starting {config['schedule']['name']}...")

    if config.get("dry_run"):
        logger.info("Dry run. Skipping execution.")
        return 0, 0.0
    run_result = run_simulator.run(
        config, lock_model_writing, list_model_writing, verbose=1
    )
    logger.info(f"Completed {config['schedule']['name']}.")
    return run_result.n_train, run_result.ml_metric


def execute_one_param(
    trial: optuna.Trial,
    config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    configure_logger(config)

    update_config_params(
        trial,
        config,
        list_drift_detector,
    )

    tscv = TimeSeriesSplit(n_splits=config["optimization_splits"]["detector"])

    test_start_idx = config["test_start_idx"]
    metrics = []
    for i, (train_index, test_index) in reversed(
        list(enumerate(tscv.split(np.arange(test_start_idx))))
    ):
        metrics.append(
            execute_one_fold(
                i,
                copy.deepcopy(config),
                train_index,
                test_index,
                lock_model_writing,
                list_model_writing,
            )
        )

    n_train = [m[0] for m in metrics]
    ml_metric = [m[1] for m in metrics]

    return float(np.mean(n_train)), float(np.mean(ml_metric))


def run(
    config: Dict[str, Any],
    lock_model_writing: Optional[LockType] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:

    # Merge best params
    if config.get("use_auto_model_tuning"):
        config = add_best_params_to_model(config)

    # Add auto trial params
    add_auto_trial_params(config)

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    logger = logging.getLogger(__name__)
    # CONFIG

    schedule_name = config["schedule"]["name"]
    logger.info(f"Optimizing schedule {config['schedule']}")

    # LOAD AND CREATE OBJECTS
    list_drift_detector = get_drift_detector_class_from_conf(
        config["schedule"]["detectors"]
    )

    data_path = config["schedule_data_path"]
    Path(data_path).mkdir(parents=True, exist_ok=True)
    studies_path = f"{data_path}/optuna_study.db"

    # warnings.filterwarnings("ignore", category=ExperimentalWarning)

    failed_trial_callback = RetryFailedTrialCallback(max_retry=None)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{studies_path}",
        heartbeat_interval=10,
        grace_period=20,
        failed_trial_callback=failed_trial_callback,
    )

    sampler_path = f"{data_path}/optuna_sampler.joblib"
    if os.path.exists(sampler_path):
        sampler = joblib.load(sampler_path)
    else:
        sampler = TPESampler(n_startup_trials=5, seed=42, multivariate=True)
        joblib.dump(
            sampler,
            sampler_path,
        )

    study = optuna.create_study(
        study_name=schedule_name,
        sampler=sampler,
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    n_to_finish = config["optimization_iter"]["detector"]

    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    logger.info(f"Completed {schedule_name}: {n_completed}")

    def logger_done_callback(
        study_l: Study, frozen_trial: FrozenTrial
    ) -> None:
        n_done = len(study_l.get_trials(states=(TrialState.COMPLETE,)))
        logger.info(f"Completed {schedule_name}: {n_done}")

    if n_completed == 0:
        default_params = get_default_params(config, list_drift_detector)
        if default_params is not None:
            study.enqueue_trial(default_params)

    if n_completed < n_to_finish:
        study.optimize(
            lambda trial_l: execute_one_param(
                trial_l,
                copy.deepcopy(config),
                list_drift_detector,
                lock_model_writing,
                list_model_writing,
            ),
            callbacks=[
                MaxTrialsCallback(
                    n_to_finish,
                    states=(TrialState.COMPLETE,),
                ),
                lambda *_: joblib.dump(sampler, sampler_path),
                logger_done_callback,
            ],
        )

    n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
    logger.info(f"Completed {schedule_name}: {n_completed}.")
    best_trials = study.best_trials
    best_trials = [b for b in best_trials if b.values[0] > 1]
    save_json(
        {
            "best_metrics": [
                {"n_train": b.values[0], "ml_metric": b.values[1]}
                for b in best_trials
            ]
        },
        f"{data_path}/best_metrics.json",
    )
    save_json(
        {"best_params": [b.params for b in best_trials]},
        f"{data_path}/best_params.json",
    )
    best_schedules = [
        load_json(f"{data_path}/schedules/iter_{b.number}_fold_0/config.json")[
            "schedule"
        ]
        for b in best_trials
    ]
    save_json(
        {"best_schedules": best_schedules},
        f"{data_path}/best_schedules.json",
    )


def run_any(config: Dict[str, Any]) -> None:
    configure_logger(config)

    if config.get("schedule") is not None:
        return run(config)

    n_jobs_optimiser = (
        config["performance"].get("n_jobs", {}).get("optimizer", 1)
    )
    logger = logging.getLogger(__name__)

    confs = []
    for i in range(len(config.get("schedules"))):
        conf = copy.deepcopy(config)
        conf["schedule"] = conf["schedules"][i]
        conf[
            "schedule_data_path"
        ] = f"{conf['schedule_data_path']}/{conf['schedule']['name']}"
        conf.pop("schedules")
        confs.append(conf)

    if n_jobs_optimiser == 1:
        logger.info("Running in sequence.")
        for c in confs:
            run(c)
    else:
        logger.info("Running in parallel.")
        with Manager() as manager:
            lock = manager.Lock()
            dico: Dict[str, Any] = manager.dict()
            with parallel_backend("loky", n_jobs=n_jobs_optimiser):
                Parallel()(delayed(run)(e, lock, dico) for e in confs)


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)
    run_any(config)
