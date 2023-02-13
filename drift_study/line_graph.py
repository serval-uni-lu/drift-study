import logging
import os

import configutils
import pandas as pd
from configutils.utils import merge_parameters
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric_factory import create_metric

from drift_study.reports.graphics import lineplot
from drift_study.utils.evaluation import load_config_eval

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run() -> None:
    config = configutils.get_config()

    dataset = get_dataset(config.get("dataset"))
    for i in range(len(config.get("runs"))):
        config.get("runs")[i] = merge_parameters(
            config.get("common_runs_params").copy(),
            config.get("runs")[i].copy(),
        )

    logger.info(f"Starting dataset {dataset.name}")
    x, y, t = dataset.get_x_y_t()

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    config = load_config_eval(config, dataset, prediction_metric, y)

    for_df = {"run_name": [], "x_batch": [], "metric": []}
    for e in config["runs"]:
        for_df["run_name"].extend([e["name"]] * len(e["batch_start_idx"]))
        for_df["x_batch"].extend(e["batch_start_idx"].tolist())
        for_df["metric"].extend(e["prediction_metric_batch"].tolist())

    df = pd.DataFrame.from_dict(for_df)
    output_file = config.get("output_file", None)

    plot_engine = config.get("plot_engine", "sns")
    if plot_engine == "sns":
        lineplot(
            df,
            output_file,
            x="x_batch",
            y="metric",
            y_label=config["evaluation_params"]["metric"]["name"].upper(),
            hue="run_name",
            x_label="\\# Input",
            fig_size=(6, 4),
            legend_pos="best",
        )
    elif plot_engine == "plotly":
        import plotly.express as px

        fig = px.line(
            df,
            x="x_batch",
            y="metric",
            color="run_name",
        )
        fig.write_html(output_file)


if __name__ == "__main__":
    run()
