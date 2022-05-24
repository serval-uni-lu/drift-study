import logging
import os

import h5py
import joblib
import numpy as np
from matplotlib import pyplot as plt
from mlc.datasets import load_datasets
from sklearn.metrics import f1_score, matthews_corrcoef

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


configs = [
    # {
    #     "dataset_name": "electricity",
    #     "model_name": "random_forest",
    #     "drift_name": "adwin",
    #     "window_size": 10000,
    # },
    # {
    #     "dataset_name": "lcld",
    #     "model_name": "random_forest",
    #     "drift_name": "adwin",
    #     "window_size": 100000,
    # },
    # {
    #     "dataset_name": "lcld",
    #     "model_name": "random_forest",
    #     "drift_name": "tabular",
    #     "window_size": 100000,
    # },
    {
        "dataset_name": "lcld",
        "model_name": "random_forest",
        "drift_name": "evidently",
        "window_size": 100000,
    }
]


def moving_average(a, n=20000):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


# def moving_average(x, w):
#     return np.convolve(x, np.ones(w), 'valid') / w


def run(config):
    print(config)
    dataset_name = config.get("dataset_name")
    model_name = config.get("model_name")
    logger.info(f"Starting dataset {dataset_name}, model {model_name}")
    dataset = load_datasets(dataset_name)
    x, y, t = dataset.get_x_y_t()

    drift_data_path = (
        f"./data/{dataset_name}/drift/{model_name}_{config.get('drift_name')}"
    )
    with h5py.File(drift_data_path, "r") as f:
        is_drifts = f["is_drifts"][()]
        is_drift_warnings = f["is_drift_warnings"][()]
        y_scores = f["y_scores"][()]
        model_used = f["model_used"][()]

    model_path = (
        f"./models/{dataset_name}/{model_name}_{config.get('drift_name')}"
    )
    fitted_models = [
        joblib.load(f"{model_path}_{i}.joblib")
        for i in np.arange(np.max(model_used))
    ]

    test_i = np.arange(len(x))[config.get("window_size") :]
    y_test_static = fitted_models[0].predict_proba(x)
    corrects_avg = []
    for eval_name, y_eval in [
        ("static", y_test_static),
        ("dynamic", y_scores),
    ]:
        y_test = y[test_i]
        y_pred = np.argmax(y_eval, axis=1)[test_i]
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        correct = y_test == y_pred
        correct_avg = moving_average(correct, 5000)
        corrects_avg.append(correct_avg)

        tp = (y_pred == 1) & (y_test == y_pred)
        fp = (y_pred == 1) & (y_test != y_pred)
        tn = (y_pred == 0) & (y_test == y_pred)
        fn = (y_pred == 0) & (y_test != y_pred)

        tpa = moving_average(tp, 20000)
        fpa = moving_average(fp, 20000)
        tna = moving_average(tn, 20000)
        fna = moving_average(fn, 20000)

        precision_a = tpa / (tpa + fpa)
        recall_a = tpa / (tpa + fna)
        f1a = 2 * (precision_a * recall_a / (precision_a + recall_a))
        mcca = (tpa * tna - fpa * fna) / np.sqrt(
            (tpa + fpa) * (tpa + fna) * (tna + fpa) * (tna + fna)
        )

        plt.plot(f1a, label=f"{eval_name}, mcc {mcc}")
        plt.plot(mcca, label=f"{eval_name}, f1 {f1}")

    plt.legend()
    plt.savefig("reports/test.pdf")
    plt.clf()

    plt.plot(model_used[test_i])
    plt.savefig("reports/model_used.pdf")
    plt.clf()

    plt.plot(is_drift_warnings[test_i], label="Warning")
    plt.plot(is_drifts[test_i], label="Drift")
    plt.legend()
    plt.savefig("reports/drifts.pdf")
    plt.clf()

    logger.info("Done.")


def run_many(configs):
    for config in configs:
        run(config)


if __name__ == "__main__":
    run_many(configs)
