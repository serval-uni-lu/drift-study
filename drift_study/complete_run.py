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
import yaml
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
    if "config" in config:
        run_name = config["config"]["runs"][0]["name"]
    else:
        run_name = config["runs"][0]["name"]

    index = re.search(r"\d", run_name).start()
    return run_name[:index]


def get_metrics(config: Dict[str, Any]) -> Tuple[float, float]:
    n_train, ml_metric = config["n_train"], config["ml_metric"]
    if isinstance(n_train, list):
        n_train = float(np.mean(n_train))
        ml_metric = float(np.mean(ml_metric))

    return n_train, ml_metric


def group_by_name(
    configs: List[Dict[str, Any]]
) -> Dict[str, npt.NDArray[np.int_]]:
    str_list = [get_run_name(e) for e in configs]
    out: Dict[str, npt.NDArray[np.int_]] = {}

    for i, e in enumerate(str_list):
        if e in out:
            out[e] = np.append(out[e], i)
        else:
            out[e] = np.array([i])

    return out


def filter_n_train(result: Dict[str, Any]):
    # No detection must be run for baseline
    if get_run_name(result) == "no_detection":
        return True

    # We do not execute the run if it never triggered retrain for any fold.
    return get_metrics(result)[0] > 1.0


def pareto_rank_by_group(
    optimize_configs: List[Dict[str, Any]]
) -> npt.NDArray[np.int_]:
    configs_group = group_by_name(optimize_configs)
    configs_rank_in_group = np.full(len(optimize_configs), -1)
    objective_direction = np.array([1, -1])

    config_metrics = np.array([list(get_metrics(e)) for e in optimize_configs])

    for group in configs_group.keys():
        group_idxs = configs_group[group]
        if group == "no_detection":
            # We want to run a single no detection, therefore
            # Pareto rank 1 for first no detection, infinity for the others.
            pareto_rank = np.array(
                [1]
                + [np.iinfo(np.int32).max]
                * (len(configs_rank_in_group[configs_group[group]]) - 1)
            )
        else:
            pareto_rank = calc_pareto_rank(
                config_metrics[group_idxs],
                objective_direction,
            )
        configs_rank_in_group[group_idxs] = pareto_rank

    return configs_rank_in_group


def filter_configs(
    config: Dict[str, Any], optimize_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    # Filter by n_train (n_train > 1 for all detectors except no detection)
    optimize_configs = list(
        filter(filter_n_train, copy.deepcopy(optimize_configs))
    )

    # Evaluate
    configs_rank_in_group = pareto_rank_by_group(optimize_configs)

    # Filter by max config
    pareto_filter = configs_rank_in_group <= int(config["max_pareto"])
    optimize_configs = [
        e for e, f in zip(optimize_configs, pareto_filter) if f
    ]

    return optimize_configs


def update_config_to_run(
    config: Dict[str, Any], optimize_config: Dict[str, Any]
) -> Dict[str, Any]:

    # Get config
    config_l = optimize_config["config"]
    config_run = config_l["runs"][0]

    # Update
    config_run["last_idx"] = -1
    config_l["n_early_stopping"] = config.get("n_early_stopping", -1)

    retraining_delay = config.get("retraining_delay")
    if retraining_delay is not None:
        config_l["common_runs_params"]["delays"][
            "retraining"
        ] = retraining_delay
        config_run["delays"]["retraining"] = retraining_delay

    return config_l


def filter_config_to_run(
    config: Dict[str, Any], optimize_configs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    configs_metrics = np.array([get_metrics(e) for e in optimize_configs])

    configs_group = group_by_name(optimize_configs)
    configs_rank_in_group = np.full(len(optimize_configs), -1)

    optimize_configs = list(
        filter(filter_n_train, copy.deepcopy(optimize_configs))
    )

    for group in configs_group.keys():
        n_train = configs_metrics[configs_group[group]][:, 0]
        range_before = [np.min(n_train), np.max(n_train)]
        if group == "no_detection":
            pareto_rank = np.array(
                [1]
                + [np.iinfo(np.int32).max]
                * (len(configs_rank_in_group[configs_group[group]]) - 1)
            )
        else:
            idx_to_run = np.where(
                np.logical_and(
                    (n_train > 1),
                    (n_train <= config["max_retrain"]),
                )
            )[0]
            pareto_rank = np.full(
                len(configs_rank_in_group[configs_group[group]]),
                np.iinfo(np.int32).max,
            )
            if len(idx_to_run) > 0:
                pareto_rank[idx_to_run] = calc_pareto_rank(
                    configs_metrics[configs_group[group]][idx_to_run],
                    np.array([1, -1]),
                )
        range_after = [np.min(n_train), np.max(n_train)]
        nb_train_before = len(pareto_rank)
        nb_train = np.sum(pareto_rank <= int(config["max_pareto"]))
        logger.debug(
            f"Group {group} range: "
            f"{nb_train_before}: {range_before} => {range_after}: {nb_train}"
        )
        configs_rank_in_group[configs_group[group]] = pareto_rank

    configs_to_run_idx = np.arange(len(optimize_configs))
    configs_to_run_idx = configs_to_run_idx[
        configs_rank_in_group <= int(config["max_pareto"])
    ]
    configs_to_run = [optimize_configs[i] for i in configs_to_run_idx]
    for i in range(len(configs_to_run)):
        config_l = configs_to_run[i]
        config_l = config_l["config"]
        config_l["evaluation_params"]["val_test_idx"] = config_l[
            "evaluation_params"
        ]["last_idx"]
        config_l["evaluation_params"]["last_idx"] = -1
        config_l["evaluation_params"]["n_early_stopping"] = -1
        config_l["sub_dir_path"] = config["sub_dir_path"]

        retraining_delay = config.get("retraining_delay")
        if retraining_delay is not None:
            config_l["common_runs_params"]["delays"][
                "retraining"
            ] = retraining_delay
            config_l["runs"][0]["delays"]["retraining"] = retraining_delay

        config_l["runs"][0]["end_train_idx"] = -1
        configs_to_run[i] = config_l

    return configs_to_run


def run_many(
    config: Dict[str, Any], configs_to_run: List[Dict[str, Any]]
) -> None:

    n_jobs_optimiser = 1

    with Manager() as manager:
        lock = manager.Lock()
        dico = manager.dict()

        Parallel(n_jobs=n_jobs_optimiser)(
            delayed(execute_one_trial)(
                configs_to_run[i], configs_to_run[i]["runs"][0], lock, dico
            )
            for i in trange(len(configs_to_run), position=0)
        )


def short_config(config: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    out = []
    for e in config:
        r = e["runs"][0]
        out.append(
            {
                "name": r["name"],
                "type": r["type"],
                "detectors": r["detectors"],
            }
        )
    return out


def run() -> None:
    config = configutils.get_config()

    optimize_configs = load_config_from_dir(config["input_dir"])
    configs_to_run = filter_configs(config, copy.deepcopy(optimize_configs))
    configs_to_run = [
        update_config_to_run(config, copy.deepcopy(e)) for e in configs_to_run
    ]
    logger.info(
        f"That would run {len(configs_to_run)} out of {len(optimize_configs)}"
    )
    # if config["do_run"]:
    #     run_many(config, configs_to_run)

    output_file = config.get("output_file")
    if config.get("output_file"):
        logger.info(f"Saving configs to {output_file}.")
        out = {
            "runs": short_config(configs_to_run),
        }
        with open(output_file, "w") as f:
            yaml.dump(out, f)


if __name__ == "__main__":
    run()
