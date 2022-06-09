import logging
import os

import configutils
import h5py
import joblib
import numpy as np
from mlc.datasets import load_datasets
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


metrics_params = {"reference_methods": ["periodic"], "significance": 0.0}


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

    significance = metrics_params.get("significance")
    for ref_config in ref_configs:
        ref_config_name = ref_config.get("drift_name")
        logger.info(f"<><><> Reference: {ref_config_name} <><><>")
        ref_f1s = ref_config.get("f1s")

        for eval_config in configs.get("runs"):
            TP, TN, FP, FN = 0, 0, 0, 0
            drift_pred = np.full(len(index_batches), np.nan)
            drift_true = np.full(len(index_batches), np.nan)
            eval_config_name = eval_config.get("run_name")
            logger.info(
                f"<><><> <><><> Config test: {eval_config_name} <><><> <><><>"
            )

            model_used = eval_config.get("model_used")
            model_path = (
                f"./models/{dataset_name}/{model_name}_{eval_config_name}"
            )
            fitted_models = [
                joblib.load(f"{model_path}_{i}.joblib")
                for i in np.arange(np.max(model_used))
            ]
            retraineds = eval_config.get("is_retrained")
            config_f1 = eval_config.get("f1s")

            for i, index_batch in enumerate(index_batches):
                if retraineds[i]:
                    if i > 0:
                        y_scores = fitted_models[
                            model_used[index_batch].max() - 1
                        ].predict_proba(x.iloc[index_batch])
                        y_pred = np.argmax(y_scores, axis=1)
                        f1_past = f1_score(y[index_batch], y_pred)
                        if (f1_past + significance) <= config_f1[i]:
                            TP += 1
                            drift_true[i] = 1
                            drift_pred[i] = 1
                        else:
                            FP += 1
                            drift_true[i] = 0
                            drift_pred[i] = 1

                else:
                    if ref_f1s[i] <= (config_f1[i] + significance):
                        TN += 1
                        drift_true[i] = 0
                        drift_pred[i] = 0
                    else:
                        FN += 1
                        drift_true[i] = 1
                        drift_pred[i] = 0

            logger.info(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
            precision = precision_score(drift_true[1:], drift_pred[1:])
            recall = recall_score(drift_true[1:], drift_pred[1:])
            logger.info(f"Precision: {precision}, " f"Recall:  {recall}")
            # logger.info(f"Should be positive {TP + FN}")
            # logger.info(f"Should be negative {TN + FP}")
            # logger.info(confusion_matrix(drift_true[1:], drift_pred[1:]))


if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
