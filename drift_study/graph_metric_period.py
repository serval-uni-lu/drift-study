import logging
import os
from pathlib import Path

import configutils
import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric_factory import create_metric
from mlc.metrics.metrics import PredClassificationMetric

from drift_study.utils.helpers import get_ref_eval_config

matplotlib.use("TkAgg")
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
font = {"size": 18}

matplotlib.rc("font", **font)


def run():
    config = configutils.get_config()
    print(config)
    ref_configs, eval_configs = get_ref_eval_config(
        config, config.get("evaluation_params").get("reference_methods")
    )

    dataset = get_dataset(config.get("dataset"))
    model_name = config.get("runs")[0].get("model").get("name")
    logger.info(f"Starting dataset {dataset.name}, model {model_name}")
    x, y, t = dataset.get_x_y_t()

    test_i = np.arange(len(x))[config.get("window_size") :]

    batch_size = config.get("evaluation_params").get("batch_size")
    length = len(test_i) - (len(test_i) % batch_size)
    index_batches = np.split(test_i[:length], length / batch_size)
    fig_folder = f"reports/{dataset.name}/"
    Path(fig_folder).mkdir(parents=True, exist_ok=True)

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    # --- For each config, collect data needed
    for run_config in config.get("runs"):
        drift_data_path = (
            f"./data/drift-study/{dataset.name}/{model_name}/"
            f"{run_config.get('name')}.hdf5"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]
            model_used = f["model_used"][()]

        # y_pred = np.argmax(y_scores, axis=1)

        # Check if retrained
        run_config["model_used"] = model_used
        run_config["is_retrained"] = []
        for i, index_batch in enumerate(index_batches):
            if i == 0:
                run_config["is_retrained"].append(True)
            else:
                if (
                    model_used[index_batch].max()
                    != model_used[index_batches[i - 1]].max()
                ):
                    run_config["is_retrained"].append(True)
                else:
                    run_config["is_retrained"].append(False)

        if isinstance(prediction_metric, PredClassificationMetric):
            y_scores = np.argmax(y_scores, axis=1)
        run_config["f1s"] = [
            prediction_metric.compute(y[index_batch], y_scores[index_batch])
            for index_batch in index_batches
        ]

    for ref_config in ref_configs:
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        plt.figure(figsize=(20, 6))

        for i, eval_config in enumerate(config.get("runs")):
            eval_config_name = eval_config.get("name")
            model_used_max = eval_config.get("model_used").max()
            eval_f1s = np.array(eval_config.get("f1s"))
            # eval_f1s = eval_f1s[:-10]
            label = f"{eval_config_name}: {model_used_max + 1}"
            plt.plot(eval_f1s, label=label, marker=markers[i % len(markers)])

        plt.legend()
        plt.xlabel("Time ordered batch")
        plt.ylabel(prediction_metric.metric_name)
        plt.savefig(
            f"{fig_folder}/"
            f"{prediction_metric.metric_name}_"
            f"batch_{batch_size}_{ref_config_name}.pdf"
        )
        plt.clf()

    for i, ref_config in enumerate(ref_configs):
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = np.array(ref_config.get("f1s"))
        ref_f1s = ref_f1s[:-10]
        plt.figure(figsize=(20, 6))
        for eval_config in config.get("runs"):
            eval_config_name = eval_config.get("name")
            model_used_max = eval_config.get("model_used").max()
            eval_f1s = np.array(eval_config.get("f1s"))
            eval_f1s = eval_f1s[:-10]
            label = f"{eval_config_name}: {model_used_max + 1}"
            plt.plot(
                eval_f1s - ref_f1s,
                label=label,
                marker=markers[i % len(markers)],
            )

        plt.legend()
        plt.xlabel("Time ordered batch")
        plt.ylabel(prediction_metric.metric_name)
        plt.savefig(
            f"{fig_folder}/"
            f"{prediction_metric.metric_name}_"
            f"batch_{batch_size}_{ref_config_name}_delta.pdf"
        )
        plt.clf()


if __name__ == "__main__":
    run()
