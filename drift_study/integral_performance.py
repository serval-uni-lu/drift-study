import logging
import os

import configutils
import h5py
import numpy as np
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from sklearn.metrics import f1_score

from drift_study.utils.evaluation import rolling_f1
from drift_study.utils.helpers import get_ref_eval_config

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def run():
    config = configutils.get_config()

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
    for run_config in config.get("runs"):
        drift_data_path = (
            f"./data/{dataset.name}/drift/"
            f"{model_name}_{run_config.get('name')}"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]
            model_used = f["model_used"][()]

        y_pred = np.argmax(y_scores, axis=1)

        run_config["model_used"] = model_used
        if config.get("evaluation_params").get("rolling"):
            run_config["f1s"] = rolling_f1(
                y[test_i], y_pred[test_i], n=batch_size
            )
        else:
            run_config["f1s"] = np.array(
                [
                    f1_score(y[index_batch], y_pred[index_batch])
                    for index_batch in index_batches
                ]
            )
    out = []
    for ref_config in ref_configs:
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = ref_config.get("f1s")

        for eval_config in config.get("runs"):
            eval_config_name = eval_config.get("name")
            config_f1 = eval_config.get("f1s")
            area = np.trapz(config_f1 - ref_f1s)
            area_scaled = area / len(config_f1)
            out.append(
                {
                    "reference": ref_config_name,
                    "eval": eval_config_name,
                    "n_train": eval_config.get("model_used").max() + 1,
                    "area": area,
                    "area_scaled": area_scaled,
                }
            )
            logger.info(f"Eval: {eval_config_name}, Area = {area_scaled}")

    out_path = f"./reports/{dataset.name}/{model_name}_integral.csv"
    out = pd.DataFrame(out)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    run()
