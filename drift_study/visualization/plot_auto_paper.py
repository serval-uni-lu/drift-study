import logging
from os import listdir, path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import configutils
import numpy as np
import pandas as pd
import plotly.express as px
from mlc.load_do_save import load_json

from drift_study.reports.naming import name_to_type, name_to_name
from drift_study.utils.logging import configure_logger
from drift_study.utils.pareto import calc_pareto_rank


def check_present(found_schedules: List[str], name: str) -> None:
    logger = logging.getLogger(__name__)
    found = any([name in e for e in found_schedules])
    if not found:
        logger.warning(f"{name} not found in schedules.")


def filter_schedules(expected: List[str], schedule: str) -> bool:
    if schedule.startswith("periodic"):
        return schedule in expected
    return any([e in schedule for e in expected])


def get_paths(config: Dict[str, Any]) -> List[str]:

    schedule_dir = Path(config["schedule_data_path"])

    expected = (
        ["no_retrain"]
        + ["manual"]
        + [f"periodic_{e}" for e in config.get("periods", [])]
        + [e["name"] for e in config.get("schedules", [])]
    )

    found_schedules = listdir(schedule_dir)
    print(found_schedules)
    found_schedules = [
        e for e in found_schedules if path.isdir(f"{schedule_dir}/{e}")
    ]
    for name in expected:
        check_present(found_schedules, name)

    filtered_schedules = list(
        filter(lambda x: filter_schedules(expected, x), found_schedules)
    )

    print(filtered_schedules)

    paths = [f"{schedule_dir}/{e}/" for e in filtered_schedules]
    return paths


def get_config_metrics(
    paths: List[str],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:

    out = []
    for p in paths:
        out.append(
            (
                load_json(f"{p}/config.json"),
                load_json(f"{p}/metrics.json"),
            )
        )
    return out


def add_rank_to_metrics(
    config_metrics: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> None:

    n_train = np.array([e[1]["n_train"] for e in config_metrics])
    ml_metric = np.array([e[1]["ml_metric"] for e in config_metrics])

    pareto_rank = calc_pareto_rank(
        np.array([n_train, ml_metric]).T, np.array([1, -1])
    )

    for i, e in enumerate(config_metrics):
        e[1]["pareto_rank"] = pareto_rank[i]

def filter_extreme(df, max_n_train):
    filter_ok = df["schedule_type"] == "Baseline"
    filter_ok = (df["n_train"] > 1) and (df["n_train"])
    
    out = [
        conf
        for conf in confs
        if (
            (conf["detector_type"] <= "baseline")
            or ((conf["n_train"] <= extreme_value) and conf["n_train"] > 1)
        )
    ]

def filter_max_pareto_rank(
    df: pd.DataFrame,
    max_pareto_rank: int = 1,
) -> pd.DataFrame:
    return df[df["pareto_rank"] <= max_pareto_rank]


def plot_ml_n_train(df: pd.DataFrame, out_path: Optional[str] = None) -> None:

    new_df = []

    for rank in range(1, df["pareto_rank"].max() + 1):

        rank_df = df[df["pareto_rank"] == rank].copy()
        for j in range(rank, df["pareto_rank"].max() + 1):
            to_add = rank_df.copy()
            to_add["display_rank"] = j
            new_df.append(to_add)

    new_df = pd.concat(new_df)
    # df_pareto
    print(new_df)
    fig = px.scatter(
        new_df,
        x="n_train",
        y="ml_metric",
        # color="pareto_rank",
        color="schedule_type",
        symbol="schedule_type",
        hover_data=["shedule_name", "pareto_rank"],
        animation_frame="display_rank",
    )
    if out_path is None:
        fig.show()
    else:
        fig.write_html(out_path)


def to_dataframe(
    config_metrics: List[Tuple[Dict[str, Any], Dict[str, Any]]]
) -> pd.DataFrame:
    prepare = {}
    prepare["n_train"] = [e[1]["n_train"] for e in config_metrics]
    prepare["ml_metric"] = [e[1]["ml_metric"] for e in config_metrics]
    prepare["pareto_rank"] = [e[1]["pareto_rank"] for e in config_metrics]
    prepare["shedule_name"] = [
        e[0]["schedule"]["name"] for e in config_metrics
    ]
    prepare["schedule_type"] = [
        name_to_type(e[0]["schedule"]["name"]) for e in config_metrics
    ]
    prepare["shedule_nice_name"] = [name_to_name(e[0]["schedule"]["name"]) for e in config_metrics]
    return pd.DataFrame(prepare)


def run(config: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running plot_auto.py")
    config_metrics = get_config_metrics(get_paths(config))
    add_rank_to_metrics(config_metrics)
    df = to_dataframe(config_metrics)
    df.to_csv(config.get("plot_path")+".csv")
    
    aggregated_df = df.groupby('shedule_nice_name').agg(
        total_count=('shedule_nice_name', 'count'),
        sum_display_rank_1=('pareto_rank', lambda x: (x == 1).sum())
    ).reset_index().to_csv(config.get("plot_path")+".agg.csv")
    plot_ml_n_train(df, out_path=config.get("plot_path"))


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)
    run(config)
