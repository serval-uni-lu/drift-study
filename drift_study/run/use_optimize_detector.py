import logging
from copy import deepcopy
from multiprocessing import Manager
from typing import Any, Dict, List

import configutils
from joblib import Parallel, delayed, parallel_backend
from mlc.load_do_save import load_json

from drift_study import run_simulator
from drift_study.run.no_retrain_baseline import add_best_params_to_model
from drift_study.utils.logging import configure_logger


def prep_conf_schedule(config: Dict[str, Any]) -> List[Dict[str, Any]]:

    if config.get("use_auto_model_tuning"):
        config = add_best_params_to_model(config)

    optimization_path = config.get("detector_optimization_path")
    schedule_name = config["schedule"]["name"]
    best_schedule_path = (
        f"{optimization_path}/{schedule_name}/best_schedules.json"
    )
    best_schedules = load_json(best_schedule_path)["best_schedules"]

    out = []
    for _, schedule in enumerate(best_schedules):
        local_conf = deepcopy(config)
        local_conf["schedule"] = schedule
        local_conf["schedule"]["name"] = local_conf["schedule"]["name"].split(
            "_fold_"
        )[0]
        local_conf["schedule_data_path"] = (
            f"{local_conf['schedule_data_path']}/"
            f"{local_conf['schedule']['name']}"
        )
        logging.getLogger().info(f"{local_conf['schedule']['name']} selected.")
        logging.getLogger().debug(f"{local_conf['schedule_data_path']}")
        out.append(local_conf)
    return out


def prep_conf_list_schedule(config: Dict[str, Any]) -> List[Dict[str, Any]]:

    if config.get("schedule") is not None:
        return prep_conf_schedule(config)

    confs = []
    for i in range(len(config.get("schedules"))):
        conf = deepcopy(config)
        conf["schedule"] = conf["schedules"][i]

        conf.pop("schedules")

        confs.extend(prep_conf_schedule(conf))

    return confs


def run(config: Dict[str, Any]) -> None:
    configure_logger(config)

    confs = prep_conf_list_schedule(config)

    logger = logging.getLogger(__name__)
    logger.info(f"Running {len(confs)} schedules.")

    n_jobs_optimiser = (
        config["performance"].get("n_jobs", {}).get("optimizer", 1)
    )

    if config.get("dry_run"):
        logger.info("Dry run.")
        return

    if n_jobs_optimiser == 1:
        logger.info("Running in sequence.")
        for c in confs:
            run_simulator.run(c)
    else:
        logger.info("Running in parallel.")
        with Manager() as manager:
            lock = manager.Lock()
            dico: Dict[str, Any] = manager.dict()
            with parallel_backend("loky", n_jobs=n_jobs_optimiser):
                Parallel()(
                    delayed(run_simulator.run)(e, lock, dico) for e in confs
                )


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)
    run(config=config)
