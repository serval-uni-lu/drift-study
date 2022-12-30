import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import configutils
import optuna
from configutils.utils import merge_parameters
from mlc.load_do_save import save_json

from drift_study import run_simulator
from drift_study.utils.drift_detector_factory import (
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
            e["detector"].define_trial_parameters(
                trial, trial_params=config["trial_params"]
            ),
            e["params"],
        )
        run_config["detectors"][i]["params"] = new_params

    config["runs"] = [run_config]


def manual_save_run(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    n_train: int,
    ml_metric: float,
):
    out = {"config": config, "n_train": n_train, "ml_metric": ml_metric}
    out_path = (
        f"./data/optuna/results/"
        f"{config['dataset']['name']}/"
        f"{run_config['model']['name']}/"
        f"{run_config['name']}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)


def execute_one_trial(
    trial: optuna.Trial,
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    list_drift_detector: List[Dict[str, Any]],
) -> Tuple[int, float]:

    update_params(trial, config, run_config, list_drift_detector)
    n_train, ml_metric = run_simulator.run(config, 0)
    manual_save_run(config, run_config, n_train, ml_metric)

    return n_train, ml_metric


def run(config, run_i) -> None:

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
    studies_path = "./data/optuna/studies.db"
    Path(studies_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{studies_path}",
        directions=["minimize", "maximize"],
    )
    study.optimize(
        lambda trial: execute_one_trial(
            trial, config, run_config, list_drift_detector
        ),
        n_trials=5,
    )


def run_many() -> None:
    config_all = configutils.get_config()

    for i in range(len(config_all.get("runs"))):
        run(config_all, i)


if __name__ == "__main__":
    run_many()
