import logging
import os

import configutils
import numpy as np
import pandas as pd
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.metric_factory import create_metric

from drift_study.utils.evaluation import load_config_eval
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

    prediction_metric = create_metric(config["evaluation_params"]["metric"])
    config = load_config_eval(
        config, dataset, model_name, prediction_metric, y
    )
    out = []
    for ref_config in ref_configs:
        ref_config_name = ref_config.get("name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_metric = ref_config.get("prediction_metric_batch")

        for eval_config in config.get("runs"):
            eval_config_name = eval_config.get("name")
            eval_metric = eval_config.get("prediction_metric_batch")
            area = np.trapz(eval_metric - ref_metric)
            area_scaled = area / len(eval_metric)
            diff = eval_metric - ref_metric
            out.append(
                {
                    "reference": ref_config_name,
                    "eval": eval_config_name,
                    "n_train": eval_config.get("model_used").max() + 1,
                    "area": area,
                    "area_scaled": area_scaled,
                    "max_diff": diff.max(),
                    "mean_diff": diff.mean(),
                    "med_diff": np.median(diff),
                    "min_diff": diff.min(),
                }
            )
            logger.info(f"Eval: {eval_config_name}, Area = {area_scaled}")

    out_path = f"./reports/{dataset.name}/{model_name}_integral.csv"
    out = pd.DataFrame(out)
    out.to_csv(out_path, index=False)


if __name__ == "__main__":
    run()