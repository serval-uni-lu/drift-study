import logging
import os
from multiprocessing import Lock, Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import configutils
import joblib
import optuna
from configutils.utils import merge_parameters
from joblib import Parallel, delayed
from mlc.load_do_save import save_json
from optuna._callbacks import MaxTrialsCallback, RetryFailedTrialCallback
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from tqdm import tqdm, trange

from drift_study import run_simulator
from drift_study.drift_detectors.drift_detector_factory import (
    get_drift_detector_class_from_conf,
)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def update_params(
    trial: optuna.Trial,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> None:

    run_config["name"] = run_config["name"] + str(trial.number)
    for i, e in enumerate(list_drift_detector):
        new_params = merge_parameters(
            e["params"],
            e["detector"].define_trial_parameters(
                trial, trial_params=config["trial_params"]
            ),
        )
        logger.debug(new_params)
        run_config["detectors"][i]["params"] = new_params

    config["runs"] = [run_config]


def manual_save_run(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    n_train: int,
    ml_metric: float,
):
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


def execute_one_trial(
    trial: optuna.Trial,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> Tuple[int, float]:

    update_params(
        trial,
        config,
        run_config,
        list_drift_detector,
    )
    n_train, ml_metric = run_simulator.run(
        config, 0, lock_model_writing, list_model_writing, verbose=0
    )
    manual_save_run(config, run_config, n_train, ml_metric)

    return n_train, ml_metric


def run(
    config: Dict[str, Any],
    run_i: int,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    logger = logging.getLogger(__name__)

    # CONFIG
    run_config = merge_parameters(
        config.get("common_runs_params"), config.get("runs")[run_i]
    )
    logger.info(f"Optimizing config {run_config.get('name')}")

    # LOAD AND CREATE OBJECTS
    list_drift_detector = get_drift_detector_class_from_conf(
        run_config.get("detectors")
    )

    study_name = run_config["name"]
    model_name = run_config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    sub_dir_path = config["sub_dir_path"]

    studies_dir = (
        f"./data/optimizer/{dataset_name}/{model_name}/" f"{sub_dir_path}/"
    )

    studies_path = f"{studies_dir}/study_{study_name}.db"
    Path(studies_path).parent.mkdir(parents=True, exist_ok=True)

    failed_trial_callback = RetryFailedTrialCallback(max_retry=None)
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{studies_path}",
        heartbeat_interval=10,
        grace_period=20,
        failed_trial_callback=failed_trial_callback,
    )

    sampler_path = f"{studies_dir}/study_{study_name}.sampler"
    if os.path.exists(sampler_path):
        sampler = joblib.load(sampler_path)
    else:
        sampler = TPESampler(n_startup_trials=5, seed=42)
        joblib.dump(
            sampler,
            sampler_path,
        )

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        storage=storage,
        directions=["minimize", "maximize"],
        load_if_exists=True,
    )
    n_to_finish = config["trial_params"]["n_trials"]
    with tqdm(total=n_to_finish, position=run_i + 1) as pbar:
        pbar.set_description(f"Processing {study_name}")
        n_completed = len(study.get_trials(states=(TrialState.COMPLETE,)))
        pbar.update(n_completed)
        if n_completed < n_to_finish:
            study.optimize(
                lambda trial_l: execute_one_trial(
                    trial_l,
                    config,
                    run_config,
                    list_drift_detector,
                    lock_model_writing,
                    list_model_writing,
                ),
                callbacks=[
                    MaxTrialsCallback(
                        config["trial_params"]["n_trials"],
                        states=(TrialState.COMPLETE,),
                    ),
                    lambda *_: joblib.dump(sampler, sampler_path),
                    lambda *_: pbar.update(1),
                ],
            )


def run_many() -> None:
    config_all = configutils.get_config()

    n_jobs_optimiser = config_all["performance"].get("n_jobs_optimiser", 1)

    if n_jobs_optimiser == 1:
        logger.info("Running in sequence.")
        for i in range(len(config_all.get("runs"))):
            run(config_all, i)
    else:
        logger.info("Running in parallel.")
        with Manager() as manager:
            lock = manager.Lock()
            dico = manager.dict()

            Parallel(n_jobs=n_jobs_optimiser)(
                delayed(run)(config_all, i, lock, dico)
                for i in trange(len(config_all.get("runs")), position=0)
            )


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    run_many()
