import glob
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


def filter_conf(conf_results: Dict[str, Any]) -> Dict[str, Any]:
    detector_name = "_".join(
        [e["name"] for e in conf_results["config"]["runs"][0]["detectors"]]
    )
    if isinstance(conf_results["ml_metric"], list):
        ml_metric = conf_results["ml_metric"][2]
    else:
        ml_metric = conf_results["ml_metric"]
    out = {
        "n_train": conf_results["n_train"],
        "ml_metric": ml_metric,
        "detector_type": conf_results["config"]["runs"][0]["type"],
        "detector_name": detector_name,
        "params": conf_results["config"]["runs"][0]["detectors"][-1].get(
            "params", "No params provided"
        ),
    }
    return out


def run() -> None:
    config = configutils.get_config()
    print(config)

    input_dir = config["input_dir"]
    output_file = config.get("output_file", None)
    n_jobs = config.get("n_jobs", 1)

    list_files = glob.glob(f"{input_dir}/*.json")
    conf_results = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: filter_conf(load_json(x)))(path)
        for path in tqdm(list_files, total=len(list_files))
    )
    df = pd.DataFrame(conf_results)

    # Add rank
    pareto_rank = calc_pareto_rank(
        np.array([df["n_train"], df["ml_metric"]]).T, np.array([1, -1])
    )
    df["pareto_rank"] = pareto_rank
    # df = df[df["pareto_rank"] <= 20]
    fig = px.scatter(
        df,
        x="n_train",
        y="ml_metric",
        color="pareto_rank",
        symbol="detector_type",
        hover_data=["detector_name"],
    )
    print(df["pareto_rank"].max())
    if output_file is not None:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(output_file)
    # fig.show()


if __name__ == "__main__":
    run()
