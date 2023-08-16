import logging
import os
from typing import Any, Dict, List

import configutils
import numpy as np
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric import Metric
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric

from drift_study.reports.graphics import lineplot
from drift_study.typing import NDFloat, NDInt, NDNumber
from drift_study.utils.logging import configure_logger
from drift_study.visualization.plot_auto import get_config_metrics, get_paths

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def get_batches(test_i: NDInt, batch_size: int) -> List[NDInt]:

    if isinstance(batch_size, int):
        length = len(test_i) - (len(test_i) % batch_size)
        index_batches = np.split(test_i[:length], int(length / batch_size))
        return index_batches


def load_config_eval(
    path: str,
    y: NDNumber,
    batches: List[NDInt],
    metric: Metric,
) -> NDFloat:
    # logger = logging.getLogger(__name__)

    y_scores = pd.read_parquet(path).to_numpy()

    if isinstance(metric, PredClassificationMetric):
        y_scores = y_scores.argmax(axis=1)

    return np.array(
        [
            metric.compute(y[index_batch], y_scores[index_batch])
            for index_batch in batches
        ]
    )


def load_all_evals(
    paths: List[str], y: NDNumber, batches: List[NDInt], metric: Metric
) -> List[NDFloat]:
    return [
        load_config_eval(f"{path}/preds.parquet", y, batches, metric)
        for path in paths
    ]


def to_df(
    config_metrics: List[Dict[str, Any]],
    batch_metrics: List[NDFloat],
    batches: List[NDInt],
) -> pd.DataFrame:

    batch_x = [e[0] for e in batches]
    dict_df = {"schedule_name": [], "x_batch": [], "metric": []}
    for config_metric, batch_metric in zip(config_metrics, batch_metrics):
        dict_df["schedule_name"].extend(
            [config_metric[0]["schedule"]["name"]] * len(batch_x)
        )
        dict_df["x_batch"].extend(batch_x)
        dict_df["metric"].extend(batch_metric.tolist())

    return pd.DataFrame.from_dict(dict_df)


def plot_trace(
    df: pd.DataFrame, out_path: str, plot_engine, metric_name
) -> None:
    legend_pos = "best"
    if plot_engine == "sns":
        lineplot(
            df,
            out_path,
            x="x_batch",
            y="metric",
            y_label=metric_name.upper(),
            hue="schedule_name",
            x_label="Evaluation batch (starting index)",
            fig_size=(6, 4),
            legend_pos=legend_pos,
        )
    elif plot_engine == "plotly":
        import plotly.express as px

        fig = px.line(
            df,
            x="x_batch",
            y="metric",
            color="schedule_name",
        )
        fig.write_html(out_path)


def run(config: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    logger.info("Running line auto.py")

    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()
    test_i = np.arange(config.get("test_start_idx"), len(x))
    batches = get_batches(test_i, config.get("evaluation").get("batch_size"))
    paths = get_paths(config)
    if len(paths) == 0:
        logger.info("No paths found")
        return
    print(paths)
    config_metrics = get_config_metrics(paths)
    dataset = get_dataset(config.get("dataset"))
    metric = create_metric(config.get("metric"))
    batch_metrics = load_all_evals(paths, y, batches, metric)

    df = to_df(config_metrics, batch_metrics, batches)

    out_path = config.get("plot_path")

    plot_trace(
        df,
        out_path,
        config.get("plot_engine", "plotly"),
        config.get("metric").get("name"),
    )


if __name__ == "__main__":
    config = configutils.get_config()
    configure_logger(config)
    run(config)
