import glob
import os
from pathlib import Path
from typing import Any, Dict

import configutils
import numpy as np
import pandas as pd
import plotly.express as px
from joblib import Parallel, delayed
from mlc.load_do_save import load_json
from tqdm import tqdm

from drift_study.utils.pareto import calc_pareto_rank


def filter_conf(conf_results: Dict[str, Any], path: str) -> Dict[str, Any]:
    detector_name = "_".join(
        [e["name"] for e in conf_results["config"]["runs"][0]["detectors"]]
    )
    if isinstance(conf_results["ml_metric"], list):
        ml_metric = conf_results["ml_metric"][2]
    else:
        ml_metric = conf_results["ml_metric"]
    out = {
        "id": os.path.split(path)[1],
        "n_train": conf_results["n_train"],
        "ml_metric": ml_metric,
        "detector_type": conf_results["config"]["runs"][0]["type"],
        "detector_name": detector_name,
        "params": conf_results["config"]["runs"][0]["detectors"][-1].get(
            "params", "No params provided"
        ),
        "delay": conf_results["config"]["runs"][0]["delays"]["retraining"],
    }
    return out


def load_configs_dir(input_dir: str) -> pd.DataFrame:
    list_files = glob.glob(f"{input_dir}/*.json")
    conf_results = Parallel(n_jobs=1)(
        delayed(lambda x: filter_conf(load_json(x), x))(path)
        for path in tqdm(list_files, total=len(list_files))
    )
    df = pd.DataFrame(conf_results)
    df = df.sort_values(by=["id"])
    df = df.reset_index(drop=True)
    return df


def run() -> None:
    config = configutils.get_config()
    print(config)

    input_dir = config["input_dir"]
    output_file = config.get("output_file", None)

    df = load_configs_dir(input_dir)

    # Add rank
    pareto_rank = calc_pareto_rank(
        np.array([df["n_train"], df["ml_metric"]]).T, np.array([1, -1])
    )
    df["pareto_rank"] = pareto_rank
    pareto_filter = df["pareto_rank"] <= 2
    df = df[pareto_filter]

    for suffix in ["half", "twice"]:
        df_l = load_configs_dir(f"{input_dir}_{suffix}")
        df_l = df_l[pareto_filter]
        df = pd.concat([df, df_l])

    df["delay"] = pd.to_timedelta(df["delay"])
    df = df.sort_values(by=["delay"])
    fig = px.line(
        df,
        x="n_train",
        y="ml_metric",
        line_group="id",
        color="detector_type",
        hover_data=["detector_name", "delay"],
    )

    # px.line()
    print(df["pareto_rank"].max())
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        # fig.write_html(output_file)
    fig.show()


if __name__ == "__main__":
    run()
