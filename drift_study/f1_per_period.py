import logging
import os

import configutils
import h5py
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mlc.datasets import load_datasets
from sklearn.metrics import f1_score

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


metrics_params = {"reference_methods": ["periodic"]}
markers = ["d", "v", "s", "*", "^", "d", "v", "s", "*", "^"]
font = {"family": "normal", "size": 18}

matplotlib.rc("font", **font)


def run(configs):
    print(configs)
    ref_configs = []
    other_configs = []
    for config in configs.get("runs"):
        if config.get("run_name") in metrics_params.get("reference_methods"):
            ref_configs.append(config)
        else:
            other_configs.append(config)
    dataset_name = ref_configs[0].get("dataset_name")
    model_name = ref_configs[0].get("model_name")
    logger.info(f"Starting dataset {dataset_name}, model {model_name}")

    dataset = load_datasets(dataset_name)
    x, y, t = dataset.get_x_y_t()

    common_params = configs.get("common_params")
    test_i = np.arange(len(x))[common_params.get("window_size") :]

    batch_size = common_params.get("batch_size")
    length = len(test_i) - (len(test_i) % batch_size)
    index_batches = np.split(test_i[:length], length / batch_size)

    for config in configs.get("runs"):
        drift_data_path = (
            f"./data/{dataset_name}/drift/"
            f"{model_name}_{config.get('run_name')}"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]
            model_used = f["model_used"][()]

        y_pred = np.argmax(y_scores, axis=1)

        config["is_retrained"] = []
        for i, index_batch in enumerate(index_batches):
            if i == 0:
                config["is_retrained"].append(True)
            else:
                if (
                    model_used[index_batch].max()
                    != model_used[index_batches[i - 1]].max()
                ):
                    config["is_retrained"].append(True)
                else:
                    config["is_retrained"].append(False)

        config["f1s"] = [
            f1_score(y[index_batch], y_pred[index_batch])
            for index_batch in index_batches
        ]
        config["model_used"] = model_used

    for ref_config in ref_configs:
        ref_config_name = ref_config.get("run_name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = np.array(ref_config.get("f1s"))
        ref_f1s = ref_f1s[:-10]
        plt.figure(figsize=(20, 6))

        for i, eval_config in enumerate(configs.get("runs")):
            eval_config_name = eval_config.get("run_name")
            model_used_max = eval_config.get("model_used").max()
            eval_f1s = np.array(eval_config.get("f1s"))
            eval_f1s = eval_f1s[:-10]
            label = f"{eval_config_name}: {model_used_max + 1}"
            plt.plot(eval_f1s, label=label, marker=markers[i % len(markers)])

        plt.legend()
        plt.xlabel("Time ordered batch")
        plt.ylabel("f1 score")
        plt.savefig(
            f"reports/{dataset_name}/"
            f"f1_batch_{batch_size}_{ref_config_name}.pdf"
        )
        plt.clf()

    for i, ref_config in enumerate(ref_configs):
        ref_config_name = ref_config.get("run_name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = np.array(ref_config.get("f1s"))
        ref_f1s = ref_f1s[:-10]
        plt.figure(figsize=(20, 6))
        for eval_config in configs.get("runs"):
            eval_config_name = eval_config.get("run_name")
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
        plt.ylabel("f1 score")
        plt.savefig(
            f"reports/{dataset_name}/"
            f"f1_delta_batch_{batch_size}_{ref_config_name}.pdf"
        )
        plt.clf()


if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
