import copy
import logging.config
from pathlib import Path
from typing import Any, Dict, List

import configutils
import numpy as np
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
        f"{d['random_state']}_"
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

            reference[build_key(r)] = {
                "y_scores": r["y_scores"],
                "prediction_metric_batch": r["prediction_metric_batch"],
                "batch_start_idx": r["batch_start_idx"],
            }
            min_start = min(min_start, r["end_train_idx"])
            count += 1

    assert len(reference.keys()) == count

    for r in runs:
        if not r["train_all_data"]:
            if min_start < r["end_train_idx"]:
                ref_key = build_key(r)
                # Update y scores
                ref_y_scores = reference[ref_key]["y_scores"]
                r["y_scores"][min_start : r["end_train_idx"]] = ref_y_scores[
                    min_start : r["end_train_idx"]
                ]

                # Update per batch preds
                idx_to_add = np.where(
                    reference[ref_key]["batch_start_idx"]
                    < r["batch_start_idx"][0]
                )[0]
                for e in ["batch_start_idx", "prediction_metric_batch"]:
                    r[e] = np.concatenate(
                        [
                            reference[ref_key][e][idx_to_add],
                            r[e],
                        ]
                    )

    return min_start


def compute_metrics(
    runs: List[Dict[str, Any]], metric: Metric, y, start_idx
) -> None:
    for r in runs:
        r["metric"] = metric.compute(y[start_idx:], r["y_scores"][start_idx:])
        r["metric_batch"] = r["prediction_metric_batch"]


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
            "metric_batch": r["metric_batch"],
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

    df_all_out = df.drop(["metric_batch"], axis=1)
    df_all_out = df_all_out.groupby(
        ["delays", "period", "window_size", "model"]
    ).agg({"metric": ["mean", "std"]})

    def text(x):
        avg = np.mean(x)
        std = np.std(x)
        std_str = std * 10000
        return f"{avg:.5f} \\footnotesize{{$\\pm$ {std_str:.1f}}}"

    pd.pivot_table(
        df_all_out,
        index=["delays", "window_size", "model"],
        columns=["period"],
        values=["metric"],
        aggfunc={
            "metric": ["mean", "std", text],
        },
    ).swaplevel(1, 2, 0).swaplevel(0, 1, 0).sort_index(
        level=0, axis=0
    ).swaplevel(
        0, 1, 1
    )[
        "text"
    ].to_csv(
        "results.csv"
    )

    for name, group in grouped:
        df_l = group[["window_size", "period", "metric", "metric_batch"]]
        print(f"##### {name}")

        def batch_min(a, *args, **kwargs):
            return np.min(np.concatenate(a.values))

        def batch_mean(a, *args, **kwargs):
            return np.mean(np.concatenate(a.values))

        def batch_std(a, *args, **kwargs):
            return np.std(np.concatenate(a.values))

        def batch_q5(a, *args, **kwargs):
            return np.quantile(np.concatenate(a.values), 0.05)

        pivot_table = (
            pd.pivot_table(
                df_l,
                index=["window_size"],
                columns=["period"],
                values=["metric", "metric_batch"],
                aggfunc={
                    "metric": ["mean", "std"],
                    "metric_batch": [
                        batch_min,
                        batch_q5,
                        batch_mean,
                        batch_std,
                    ],
                },
            )
            .droplevel(0, 1)
            .swaplevel(0, 1, 1)
            .sort_index(level=0, axis=1)
        )
        out_path = f"reports/periodic/{dataset.name}/{name}.csv"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pivot_table.to_csv(out_path)

        print(pivot_table)
        pivot_table = pivot_table.values
        print(f"Max col in row {np.argmax(pivot_table, axis =0)}")
        print(f"Max row in col {np.argmax(pivot_table, axis =1)}")
        max_idx = (
            np.argmax(pivot_table) // pivot_table.shape[1],
            np.argmax(pivot_table) % pivot_table.shape[1],
        )
        print(f"Max index {max_idx}")


if __name__ == "__main__":
    config_all = configutils.get_config()
    configure_logger(config_all)
    run(config_all)
