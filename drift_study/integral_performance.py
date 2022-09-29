import logging
import os

import configutils
import h5py
import numpy as np
from mlc.datasets import load_datasets
from sklearn.metrics import f1_score

from drift_study.utils.evaluation import rolling_f1
from drift_study.utils.helpers import get_ref_eval_config

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run(configs):
    print(configs)

    evaluation_params = configs.get("evaluation_params")

    ref_configs, eval_configs = get_ref_eval_config(
        configs, configs.get("evaluation_params").get("reference_methods")
    )
    dataset_name = ref_configs[0].get("dataset_name")
    model_name = ref_configs[0].get("model_name")
    logger.info(f"Starting dataset {dataset_name}, model {model_name}")

    dataset = load_datasets(dataset_name)
    x, y, t = dataset.get_x_y_t()

    common_params = configs.get("common_params")
    test_i = np.arange(len(x))[common_params.get("window_size") :]
    batch_size = evaluation_params.get("batch_size")
    length = len(test_i) - (len(test_i) % batch_size)
    index_batches = np.split(test_i[:length], length / batch_size)
    for config in configs.get("runs"):
        drift_data_path = (
            f"./data/{dataset_name}/drift/"
            f"{model_name}_{config.get('run_name')}"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]

        y_pred = np.argmax(y_scores, axis=1)

        if evaluation_params.get("rolling"):
            config["f1s"] = rolling_f1(y[test_i], y_pred[test_i], n=batch_size)
        else:
            config["f1s"] = np.array(
                [
                    f1_score(y[index_batch], y_pred[index_batch])
                    for index_batch in index_batches
                ]
            )

    for ref_config in ref_configs:
        ref_config_name = ref_config.get("run_name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = ref_config.get("f1s")

        for eval_config in configs.get("runs"):
            eval_config_name = eval_config.get("run_name")
            config_f1 = eval_config.get("f1s")
            area = np.trapz(config_f1 - ref_f1s) / len(config_f1)
            logger.info(f"Eval: {eval_config_name}, Area = {area}")


if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
