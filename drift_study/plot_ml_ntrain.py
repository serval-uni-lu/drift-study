import glob
from pathlib import Path
from typing import Any, Dict

import configutils
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mlc.load_do_save import load_json
from tqdm import tqdm

from drift_study.utils.graphics import scatterplot
from drift_study.utils.naming import beautify_dataframe
from drift_study.utils.pareto import calc_pareto_rank


def filter_conf(conf_results: Dict[str, Any]) -> Dict[str, Any]:
    detector_name = "_".join(
        [e["name"] for e in conf_results["config"]["runs"][0]["detectors"]]
    )
    metric_name = (
        conf_results["config"]["runs"][0]["detectors"][-1]
        .get("params", {})
        .get("metric_conf", {})
        .get("name")
    )
    if metric_name is not None:
        detector_name = f"{detector_name}_{metric_name}"
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
        "metric_name": conf_results["config"]["evaluation_params"]["metric"][
            "name"
        ],
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

    df = beautify_dataframe(df.copy())

    plot_engine = config.get("plot_engine", "sns")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    if plot_engine == "sns":

        scatterplot(
            df,
            output_file,
            x="n_train",
            y="ml_metric",
            y_label=conf_results[0]["metric_name"].upper(),
            hue="Type",
            x_label="\\# Train",
            fig_size=(6, 4),
            legend_pos="best",
            markers=["o", "s", "^", "x"],
        )
    elif plot_engine == "plotly":
        import plotly.express as px

        fig = px.scatter(
            df,
            x="n_train",
            y="ml_metric",
            color="pareto_rank",
            symbol="detector_type",
            hover_data=["detector_name"],
        )
        fig.write_html(output_file)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    run()
