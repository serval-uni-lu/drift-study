import argparse
import logging

import numpy as np
import pandas as pd
import torch
from constrained_attacks.attacks.cta.cfab import CFAB
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


def generate_attack(dataset, model, scaler):
    logger = logging.getLogger(__name__)
    dataset_name = dataset.name
    x, y = dataset.get_x_y()
    x_test = x.iloc[splits[dataset_name]["val"] :].to_numpy()
    y_test = y[splits[dataset_name]["val"] :]

    constraints = Constraints(
        feature_types=np.array(["real"] * x.shape[1]),
        mutable_features=np.array([True] * x.shape[1]),
        lower_bounds=scaler.x_min,
        upper_bounds=scaler.x_max,
        relation_constraints=None,
        feature_names=None,
    )

    attack = CFAB(
        constraints,
        scaler,
        model.wrapper_model,
        model.predict_proba,
        norm="L2",
        eps=1,
        steps=100,
        n_restarts=10,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        verbose=False,
        seed=0,
        multi_targeted=False,
        n_classes=2,
        fix_equality_constraints_end=False,
        fix_equality_constraints_iter=False,
        eps_margin=0.01,
    )
    x_orig, y_orig = (
        torch.from_numpy(x_test).float()[:N_ATTACKS],
        torch.from_numpy(y_test).float()[:N_ATTACKS],
    )
    y_orig_pred = torch.argmax(
        torch.from_numpy(model.predict_proba(x_orig)), 1
    ).float()
    x_adv = (
        attack(
            x_orig.to("cuda"),
            y_orig_pred.to("cuda"),
        )
        .cpu()
        .detach()
        .numpy()
    )
    logger.info("Finished generating attack.")
    logger.info(
        f"Attack AUC: {compute_metric(model, create_metric('auc'), x_adv, y_orig)}"
    )
    logger.info(f"Number of attacks: {x_adv.shape[0]}")
    return x_adv


def save_attack(dataset, x_adv, path):
    df = pd.DataFrame(
        x_adv, columns=dataset.get_metadata(only_x=True)["feature"]
    )
    df.to_parquet(path)


def compute_distance(x, x_adv):
    return np.linalg.norm(x - x_adv, axis=1)


def run(
    dataset_name: str, model_name: str, adv_training: bool, surrogate_name: str
):
    configure_logger()
    dataset = get_dataset(dataset_name)
    x, y, t = dataset.get_x_y_t()

    x_train, x_val, x_test = (
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
    logger = logging.getLogger(__name__)
    for n, e in [("x_train", x_train), ("x_val", x_val), ("x_test", x_test)]:
        logger.info(f"Shape of {n}: {e.shape}")

    model, scaler = load_model(
        dataset,
        model_name,
        path=(
            f"models/"
            f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.model"
        ),
    )
    test_model(
        model,
        x_test,
        y_test,
    )
    x_adv = generate_attack(dataset, model, scaler)
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
    save_attack(
        dataset,
        x_adv,
        f"x_adv_"
        f"{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.parquet",
    )


if __name__ == "__main__":
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
