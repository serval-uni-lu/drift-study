import logging
from copy import deepcopy
from multiprocessing import Manager
from typing import Any, Dict

import configutils
from configutils.utils import ConfigFileParser, merge_parameters
from joblib import Parallel, delayed, parallel_backend

from drift_study import run_simulator
from drift_study.run.no_retrain_baseline import add_best_params_to_model
from drift_study.utils.logging import configure_logger
from mlc.logging.setup import delayed_with_logging


def create_config_params(
    config: Dict[str, Any], period: int
) -> Dict[str, Any]:
    config["schedule"]["detectors"][0]["params"] = {"period": period}
    config["schedule"]["name"] = f"{config['schedule']['name']}_{period}"
    config[
        "schedule_data_path"
    ] = f"{config['schedule_data_path']}/{config['schedule']['name']}"
    return config


def run(config: Dict[str, Any]) -> None:

    configure_logger(config)
    logger = logging.getLogger(__name__)

    # Merge the auto config
    config = merge_parameters(
        config, ConfigFileParser().do("./config/auto/periodic.yaml")
    )
    # If we want to use the best parameters found by the auto tuner

    if config.get("use_auto_model_tuning"):
        config = add_best_params_to_model(config)

    config_to_run = [
        create_config_params(deepcopy(config), period)
        for period in config.get("periods", [])
    ]
    n_jobs_optimiser = (
        config["performance"].get("n_jobs", {}).get("periodic", 1)
    )

    logger.info(
        f"Running {len(config_to_run)} periodic schedules "
        f"on {n_jobs_optimiser} processes."
    )

    if config.get("dry_run"):
        logger.info("Dry run, not running anything.")
        return None

    if n_jobs_optimiser == 1:
        logger.info("Running in sequence.")
        for c in config_to_run:
            run_simulator.run(c)
    else:
        logger.info("Running in parallel.")
        with Manager() as manager:
            lock = manager.Lock()
            dico: Dict[str, Any] = manager.dict()
            with parallel_backend("loky", n_jobs=n_jobs_optimiser):
                Parallel()(
                    delayed_with_logging(run_simulator.run)(e, lock, dico)
                    for e in config_to_run
                )


if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
