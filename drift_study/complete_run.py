import copy
import logging
import os
import re
from multiprocessing import Lock, Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import configutils
import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from mlc.load_do_save import load_json, save_json
from tqdm import trange

from drift_study import run_simulator
from drift_study.utils.pareto import calc_pareto_rank

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def manual_save_run(
    config: Dict[str, Any], n_train: int, ml_metric: float, out_path: str
):
    out = {"config": config, "n_train": n_train, "ml_metric": ml_metric}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_json(out, out_path)


def execute_one_trial(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))

    model_name = run_config["model"]["name"]
    dataset_name = config["dataset"]["name"]
    sub_dir_path = config["sub_dir_path"]

    out_path = (
        f"./data/complete_run/"
        f"{dataset_name}/{model_name}/"
        f"{sub_dir_path}/{run_config['name']}.json"
    )

    if not os.path.exists(out_path):
        n_train, ml_metric = run_simulator.run(
            config, 0, lock_model_writing, list_model_writing, verbose=0
        )
        manual_save_run(config, n_train, ml_metric, out_path)


def load_config_from_dir(input_dir: str) -> List[Dict[str, Any]]:
    json_files = [
        path for path in os.listdir(input_dir) if path.endswith(".json")
    ]

    optimizer_results = [
        load_json(f"{input_dir}/{path}") for path in json_files
    ]

    return optimizer_results


def get_run_name(config: Dict[str, Any]) -> str:
    run_name = config["config"]["runs"][0]["name"]
    index = re.search(r"\d", run_name).start()
    return run_name[:index]


def get_metrics(config: Dict[str, Any]) -> Tuple[float, float]:
    return config["n_train"], config["ml_metric"]


def group_by_idx(str_list: List[str]) -> Dict[str, npt.NDArray[np.int_]]:

    out: Dict[str, npt.NDArray[np.int_]] = {}

    for i, e in enumerate(str_list):
        if e in out:
            out[e] = np.append(out[e], i)
        else:
            out[e] = np.array([i])

    return out


def filter_config_to_run(
    config: Dict[str, Any], optimize_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    configs_run_name = [get_run_name(e) for e in optimize_configs]
    configs_metrics = np.array([get_metrics(e) for e in optimize_configs])

    configs_group = group_by_idx(configs_run_name)
    configs_rank_in_group = np.full(len(optimize_configs), -1)

    for group in configs_group.keys():
        pareto_rank = calc_pareto_rank(
            configs_metrics[configs_group[group]], np.array([1, -1])
        )
        if group == "no_detection":
            pareto_rank = np.array([1] + [np.inf] * (len(pareto_rank) - 1))
        else:
            idx_no_run = configs_metrics[configs_group[group]][:, 0] <= 1
            pareto_rank[idx_no_run] = np.iinfo(np.int32).max
            idx_no_run = (
                configs_metrics[configs_group[group]][:, 0]
                >= config["max_retrain"]
            )
            pareto_rank[idx_no_run] = np.iinfo(np.int32).max

        logger.debug(
            f"Group {group}: "
            f"{np.sum(pareto_rank <= int(config['max_pareto']))}"
        )
        configs_rank_in_group[configs_group[group]] = pareto_rank

    configs_to_run_idx = np.arange(len(optimize_configs))
    configs_to_run_idx = configs_to_run_idx[
        configs_rank_in_group <= int(config["max_pareto"])
    ]
    configs_to_run = [optimize_configs[i] for i in configs_to_run_idx]
    for i in range(len(configs_to_run)):
        config_l = configs_to_run[i]
        print(config_l["n_train"])
        config_l = config_l["config"]
        config_l["evaluation_params"]["last_idx"] = -1
        config_l["sub_dir_path"] = config["sub_dir_path"]
        configs_to_run[i] = config_l

    return configs_to_run


def run_many(
    config: Dict[str, Any], configs_to_run: List[Dict[str, Any]]
) -> None:

    n_jobs_optimiser = config["performance"].get("n_jobs_optimiser", 1)

    with Manager() as manager:
        lock = manager.Lock()
        dico = manager.dict()

        Parallel(n_jobs=n_jobs_optimiser)(
            delayed(execute_one_trial)(
                configs_to_run[i], configs_to_run[i]["runs"][0], lock, dico
            )
            for i in trange(len(configs_to_run), position=0)
        )


def run() -> None:
    config = configutils.get_config()

    optimize_configs = load_config_from_dir(config["input_dir"])
    configs_to_run = filter_config_to_run(
        config, copy.deepcopy(optimize_configs)
    )
    logger.info(
        f"That would run {len(configs_to_run)} out of {len(optimize_configs)}"
    )
    if config["do_run"]:
        run_many(config, configs_to_run)


if __name__ == "__main__":
    run()
