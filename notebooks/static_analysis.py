import argparse
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from constrained_attacks.attacks.cta.cfab import CFAB
from matplotlib import pyplot as plt
from mlc.constraints.constraints import Constraints
from mlc.dataloaders import get_custom_dataloader
from mlc.datasets.dataset_factory import get_dataset
from mlc.metrics.compute import compute_metric
from mlc.metrics.metric_factory import create_metric
from mlc.models.model_factory import get_model
from mlc.transformers.tab_scaler import TabScaler

from drift_study.utils.logging import configure_logger

from .test_log import test_log

splits = {
    "lcld_201317_ds_time": {"train": 380000, "val": 400000},
    "electricity": {"train": 8000, "val": 10000},
}
N_ATTACKS = 1000000
window_size = {"lcld_201317_ds_time": 5000, "electricity": 500}
window_size_ce = {"lcld_201317_ds_time": 20000, "electricity": 2000}


def load_model(
    dataset,
    model_name: str,
    path: str,
):
    logger = logging.getLogger(__name__)

    dataset_name = dataset.name
    x, _, _ = dataset.get_x_y_t()
    x_train = x.iloc[: splits[dataset_name]["train"]].to_numpy()
    scaler = TabScaler(one_hot_encode=True)
    scaler.fit(torch.from_numpy(x_train).float())

    model_arch = get_model(model_name)
    model = model_arch.load_class(
        path,
        scaler=scaler,
        x_metadata=dataset.get_metadata(only_x=True),
    )

    logger.info("Model loaded.")
    return model, scaler


def test_model(
    model,
    x,
    y,
    test_name: str = "Clean",
):
    logger = logging.getLogger(__name__)

    x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()

    metrics = compute_metric(
        model, [create_metric("auc"), create_metric("accuracy")], x, y
    )
    auc, acc = metrics[0], metrics[1]

    logger.info("Finished testing.")
    logger.info(f"{test_name} AUC: {auc}")
    logger.info(f"{test_name} Accuracy: {acc}")


def load_attack(path):
    return pd.read_parquet(path).to_numpy()


def compute_distance(scaler, x, x_adv, y, y_pred, with_sign=True):
    if with_sign:
        sign = (y == y_pred).astype(float) * 2 - 1
    else:
        sign = 1

    x = scaler.transform(x)
    x_adv = scaler.transform(x_adv)
    return sign * np.linalg.norm(x - x_adv, axis=1)


def compute_perturbation_rate(x, x_adv):
    close = np.isclose(x, x_adv, atol=1e-5)
    return np.mean(close, axis=0)


def plot_perturbation_rate(x_metadata, perturbation_rate, path):
    df = pd.DataFrame()
    df["feature"] = x_metadata.copy()["feature"]
    df["rate"] = perturbation_rate
    sns.barplot(x="feature", y="rate", data=df)
    plt.xlabel("Feature")
    plt.ylabel("Perturbation rate")
    # Turn ticks on x axis by 45 degrees with respect to the horizontal.
    plt.xticks(rotation=45, ha="right")

    plt.savefig(
        f"notebooks/fig/perturbation_rate_{path}.pdf", bbox_inches="tight"
    )
    plt.clf()


def plot_distance_distribution(distance, success_rate, path):
    sns.boxplot(distance, showfliers=False)
    plt.xlabel("Distance distribution")
    plt.ylabel("Distance (L2)")
    plt.title(f"Success rate: {success_rate}")
    plt.savefig(
        f"notebooks/fig/distance_distribution_{path}.pdf", bbox_inches="tight"
    )
    plt.clf()


def compute_avg_pertubation_size(scaler, x, x_adv):
    local_scaler = TabScaler(one_hot_encode=False)
    local_scaler.fit_scaler_data(scaler.get_scaler_data())

    x = local_scaler.transform(x)
    x_adv = local_scaler.transform(x_adv)
    return np.mean(np.abs(x - x_adv), axis=0)


def plot_avg_pertubation_size(x_metadata, avg_perturbation_size, path):
    df = pd.DataFrame()
    df["feature"] = x_metadata.copy()["feature"]
    df["avg_perturbation_size"] = avg_perturbation_size
    sns.barplot(x="feature", y="avg_perturbation_size", data=df)
    plt.xlabel("Feature")
    plt.ylabel("Average perturbation size (scaled)")
    # Turn ticks on x axis by 45 degrees with respect to the horizontal.
    plt.xticks(rotation=45, ha="right")

    plt.savefig(
        f"notebooks/fig/avg_perturbation_size_{path}.pdf", bbox_inches="tight"
    )
    plt.clf()


def compute_batch_distance(distance, batch_size):
    max_size = batch_size * (distance.shape[0] // batch_size)

    distance = distance[:max_size]
    distance = distance.reshape(-1, batch_size)
    distance = np.mean(distance, axis=1)
    return distance


def compute_batch_metric(model, metric, x, y, batch_size):
    max_size = batch_size * (x.shape[0] // batch_size)

    x, y = x[:max_size], y[:max_size]
    x = x.reshape(-1, batch_size, x.shape[1])
    y = y.reshape(-1, batch_size)

    out = []
    for i in range(x.shape[0]):
        out.append(compute_metric(model, metric, x[i], y[i]))
    return np.array(out)


def plot_scatter_correlation(x, y, path, metric_name, test_name):
    corr = np.corrcoef(x, y)
    sns.scatterplot(x=x, y=y)
    plt.xlabel(metric_name)
    plt.ylabel(test_name)
    plt.title(f"Correlation: {corr[0, 1]}")
    plt.savefig(
        f"notebooks/fig/scatter_correlation_{path}.pdf", bbox_inches="tight"
    )
    plt.clf()


def plot_coevolution(x_s, path, metric_names):
    if len(x_s) != len(metric_names):
        raise ValueError("x_s and metric_names must have the same length")

    df = pd.DataFrame()
    for i, x in enumerate(x_s):
        df_local = pd.DataFrame()
        df_local["batch"] = np.arange(x.shape[0])
        df_local["metric"] = x
        df_local["metric_name"] = metric_names[i]
        df = pd.concat([df, df_local])

    sns.lineplot(x="batch", y="metric", hue="metric_name", data=df)
    plt.xlabel("Batch")
    plt.savefig(f"notebooks/fig/evolution_{path}.pdf", bbox_inches="tight")
    plt.clf()


def run(
    dataset_name: str, model_name: str, adv_training: bool, surrogate_name: str
):
    logger = logging.getLogger(__name__)
    dataset = get_dataset(dataset_name)
    x, y, t = dataset.get_x_y_t()

    dataset_name = dataset.name
    _, _, x_test = (
        x.iloc[: splits[dataset_name]["train"]].to_numpy(),
        x.iloc[
            splits[dataset_name]["train"] : splits[dataset_name]["val"]
        ].to_numpy(),
        x.iloc[splits[dataset_name]["val"] :].to_numpy(),
    )
    _, _, y_test = (
        y[: splits[dataset_name]["train"]],
        y[splits[dataset_name]["train"] : splits[dataset_name]["val"]],
        y[splits[dataset_name]["val"] :],
    )

    model, scaler = load_model(
        dataset,
        model_name,
        path=(
            f"models/"
            f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.model"
        ),
    )
    x_adv = load_attack(
        f"x_adv_"
        f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.parquet",
    )
    test_model(
        model,
        x_test[:N_ATTACKS],
        y_test[:N_ATTACKS],
        test_name="Clean subset",
    )
    test_model(
        model,
        x_adv,
        y_test[:N_ATTACKS],
        test_name="Adversarial",
    )

    y_pred = model.predict(x_test[:N_ATTACKS])
    y_pred_adv = model.predict(x_adv)

    distance_label = compute_distance(
        scaler, x_test[:N_ATTACKS], x_adv, y_test[:N_ATTACKS], y_pred
    )

    distance_nolabel = np.abs(distance_label)

    perturbation_rate = compute_perturbation_rate(x_test[:N_ATTACKS], x_adv)
    plot_perturbation_rate(
        dataset.get_metadata(only_x=True),
        perturbation_rate,
        path=f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}",
    )
    avg_perturbation_size = compute_avg_pertubation_size(
        scaler, x_test[:N_ATTACKS], x_adv
    )
    plot_avg_pertubation_size(
        dataset.get_metadata(only_x=True),
        avg_perturbation_size,
        path=f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}",
    )

    batch_metric = compute_batch_metric(
        model,
        create_metric("mcc"),
        x_test[:N_ATTACKS],
        y_test[:N_ATTACKS],
        batch_size=window_size[dataset_name],
    )
    delta_metric = batch_metric[1:] - batch_metric[:-1]

    batch_metric_ce = compute_batch_metric(
        model,
        create_metric("mcc"),
        x_test[:N_ATTACKS],
        y_test[:N_ATTACKS],
        batch_size=window_size_ce[dataset_name],
    )
    delta_metric_ce = batch_metric_ce[1:] - batch_metric_ce[:-1]

    y_pred_proba = model.predict_proba(x_test[:N_ATTACKS])
    confidence_label = y_pred_proba[:, 0]
    confidence_label[y_test[:N_ATTACKS] == 1] = y_pred_proba[:, 1][
        y_test[:N_ATTACKS] == 1
    ]

    confidence_nolabel = y_pred_proba[:, 0]
    confidence_nolabel[y_pred[:N_ATTACKS] == 1] = y_pred_proba[:, 1][
        y_pred[:N_ATTACKS] == 1
    ]

    out = []
    for distance_name, distance in zip(
        [
            "distance_label",
            "distance_nolabel",
            "confidence_label",
            "confidence_nolabel",
        ],
        [
            distance_label,
            distance_nolabel,
            confidence_label,
            confidence_nolabel,
        ],
    ):
        # Plot

        plot_distance_distribution(
            distance,
            success_rate=np.mean(y_pred != y_pred_adv),
            path=f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}_{distance_name}",
        )

        batch_distance = compute_batch_distance(
            distance, batch_size=window_size[dataset_name]
        )

        delta_distance = batch_distance[1:] - batch_distance[:-1]
        plot_scatter_correlation(
            batch_distance,
            batch_metric,
            path=f"distance_mcc_{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}_{distance_name}",
            metric_name="Distance (L2)",
            test_name="MCC",
        )

        plot_scatter_correlation(
            delta_distance,
            delta_metric,
            path=f"distance_mcc_delta_{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}_{distance_name}",
            metric_name="Delta Distance (L2)",
            test_name="Delta MCC",
        )

        batch_distance_ce = compute_batch_distance(
            distance, batch_size=window_size_ce[dataset_name]
        )

        delta_distance_ce = batch_distance_ce[1:] - batch_distance_ce[:-1]
        plot_coevolution(
            [batch_metric_ce, batch_distance_ce],
            path=f"evolution_distance_{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}_{distance_name}",
            metric_names=["MCC", "Distance (L2)"],
        )
        plot_coevolution(
            [delta_metric_ce, delta_distance_ce],
            path=f"evolution_distance_delta_{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}_{distance_name}",
            metric_names=["Delta MCC", "Delta Distance (L2)"],
        )

        dist_mcc_corr = np.corrcoef(batch_distance, batch_metric)[0, 1]
        delta_dist_delta_mcc_corr = np.corrcoef(delta_distance, delta_metric)[
            0, 1
        ]
        for variation, measure in zip(
            [False, True], [dist_mcc_corr, delta_dist_delta_mcc_corr]
        ):
            out.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "adv_training": adv_training,
                    "distance": distance_name,
                    "variation": variation,
                    "corr": measure,
                }
            )

    out = pd.DataFrame(out)
    out.to_csv(
        f"notebooks/metrics/correlation_{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.csv",
        index=False,
    )


if __name__ == "__main__":
    configure_logger()

    parser = argparse.ArgumentParser(description="Train a basic model.")

    # Add positional arguments
    parser.add_argument("--dataset", help="Dataset name.")
    parser.add_argument("--model", help="Model name.")
    parser.add_argument(
        "--adv_training", help="Madry adversarial training? 0 - No, 1 - Yes."
    )

    args = parser.parse_args()

    # Access the parsed arguments
    dataset = args.dataset
    model = args.model
    adv_training = args.adv_training != "0"
    run(dataset, model, adv_training, model)
