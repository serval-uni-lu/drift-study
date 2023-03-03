import copy
import logging.config
from typing import Any, Dict, List

import configutils
import pandas as pd
from configutils.utils import merge_parameters
from mlc.metrics.metric import Metric
from mlc.metrics.metric_factory import create_metric

from drift_study.utils.evaluation import load_config_eval
from drift_study.utils.helpers import initialize
from drift_study.utils.logging import configure_logger

BIG_NUMBER = 1_000_000_000


def build_key(d: Dict[str, Any]):
    return (
        f"{d['model']['name']}_"
        f"{d['detectors'][0]['params']['period']}_"
        f"{d['delays']['label']}_{d['delays']['retraining']}"
    )


def common_evaluation_correction(runs: List[Dict[str, Any]]) -> int:

    reference = {}
    min_start = BIG_NUMBER

    # First save the scores and the min start
    count = 0
    for r in runs:
        if r["train_all_data"]:
            reference[build_key(r)] = r["y_scores"]
            min_start = min(min_start, r["end_train_idx"])
            count += 1

    assert len(reference.keys()) == count

    for r in runs:
        if not r["train_all_data"]:
            if min_start < r["end_train_idx"]:
                ref_y_scores = reference[build_key(r)]

                r["y_scores"][min_start : r["end_train_idx"]] = ref_y_scores[
                    min_start : r["end_train_idx"]
                ]

    return min_start


def compute_metrics(
    runs: List[Dict[str, Any]], metric: Metric, y, start_idx
) -> None:
    for r in runs:
        r["metric"] = metric.compute(y[start_idx:], r["y_scores"][start_idx:])


def build_df(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    filtered_runs: List[Dict[str, Any]] = []
    for r in runs:
        filtered_r = {
            "delays": f"{r['delays']['label']} + {r['delays']['retraining']}",
            "period": r["detectors"][0]["params"]["period"],
            "model": r["model"]["name"],
            "metric": r["metric"],
            "train_all_data": r["train_all_data"],
            "window_size": r["end_train_idx"],
        }
        filtered_runs.append(filtered_r)
    return pd.DataFrame(filtered_runs)


def run(
    config: Dict[str, Any],
) -> None:
    configure_logger(config)
    logger = logging.getLogger(__name__)

    for i in range(len(config.get("runs"))):
        config.get("runs")[i] = merge_parameters(
            copy.deepcopy(config.get("common_runs_params")),
            config.get("runs")[i].copy(),
        )

    dataset, _, _, _, y, _ = initialize(config, config["runs"][0])
    logger.info(f"Starting dataset {dataset.name}")

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    config = load_config_eval(config, dataset, prediction_metric, y)
    start_test_idx = common_evaluation_correction(config["runs"])
    compute_metrics(config["runs"], prediction_metric, y, start_test_idx)
    df = build_df(config["runs"])
    logger.info("Done")

    df["window_size"][df["train_all_data"]] = BIG_NUMBER
    df = df.drop("train_all_data", axis=1)
    grouped = df.groupby(["delays", "model"])

    for name, group in grouped:
        df_l = group[["window_size", "period", "metric"]]
        print(f"##### {name}")
        print(
            pd.pivot_table(
                df_l,
                index=["window_size"],
                columns=["period"],
                values=["metric"],
                aggfunc="mean",
            )
        )


if __name__ == "__main__":
    config_all = configutils.get_config()
    configure_logger(config_all)
    run(config_all)
