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


def train_model(
    dataset, model_name: str, adv_training: bool, x_train, y_train
):
    logger = logging.getLogger(__name__)
    dataset_name = dataset.name
    args_model = {
        "torchrln": {
            "name": "torchrln",
            "objective": "classification",
            "batch_size": 1024,
            "epochs": 100,
            "early_stopping_rounds": 100,
            "val_batch_size": 2048,
            "class_weight": "balanced",
            "metrics": ["auc"],
            "num_classes": 2,
            "n_layers": 2,
            "hidden_dim": 10,
        },
        "vime": {
            "name": "vime",
            "objective": "classification",
            "batch_size": 1024,
            "epochs": 100,
            "early_stopping_rounds": 100,
            "val_batch_size": 2048,
            "class_weight": "balanced",
            "num_classes": 2,
        },
        "tabtransformer": {
            "name": "tabtransformer",
            "objective": "classification",
            "batch_size": 1024,
            "epochs": 100,
            "early_stopping_rounds": 100,
            "val_batch_size": 2048,
            "class_weight": "balanced",
            "num_classes": 2,
            "learning_rate": -6,
        },
        "stg": {
            "name": "stg",
            "objective": "classification",
            "batch_size": 1024,
            "epochs": 100,
            "early_stopping_rounds": 100,
            "val_batch_size": 2048,
            "class_weight": "balanced",
            "num_classes": 2,
            "learning_rate": 0.003,
        },
        "tabnet": {
            "name": "tabnet",
            "objective": "classification",
            "batch_size": 1024,
            "epochs": 100,
            "early_stopping_rounds": 100,
            "val_batch_size": 2048,
            "class_weight": "balanced",
            "num_classes": 2,
        },
    }

    scaler = TabScaler(one_hot_encode=True)
    scaler.fit(torch.from_numpy(x_train).float())

    model_arch = get_model(model_name)
    # model = model_arch(scaler, x_metadata=dataset.get_metadata(only_x=True), batch_size=1024, epochs=5, early_stopping_rounds=5, num_classes=2, n_layers=10, hidden_dim=5,class_weight="balanced")
    model = model_arch(
        scaler=scaler,
        x_metadata=dataset.get_metadata(only_x=True),
        **args_model.get(model_name, {}),
    )
    logger.debug(f"Model.objective: {model.objective}")
    data_loader = None
    if adv_training:
        data_loader = get_custom_dataloader(
            "madry",
            dataset,
            model,
            scaler,
            {},
            verbose=1,
            x=x_train,
            y=y_train,
            train=True,
            batch_size=1024,
        )
    x, y = dataset.get_x_y()
    x_train, x_val, x_test = (
        x.iloc[: splits[dataset_name]["train"]].to_numpy(),
        x.iloc[
            splits[dataset_name]["train"] : splits[dataset_name]["val"]
        ].to_numpy(),
        x.iloc[splits[dataset_name]["val"] :].to_numpy(),
    )
    y_train, y_val, y_test = (
        y[: splits[dataset_name]["train"]],
        y[splits[dataset_name]["train"] : splits[dataset_name]["val"]],
        y[splits[dataset_name]["val"] :],
    )

    model.fit(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(np.array([1 - y_train, y_train]).T),
        torch.from_numpy(x_val).float(),
        torch.from_numpy(np.array([1 - y_val, y_val]).T).float(),
        custom_train_dataloader=data_loader,
    )
    logger.info("Finished training.")
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


def save_model(model, path):
    model.save(path)


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
    y_train, y_val, y_test = (
        y[: splits[dataset_name]["train"]],
        y[splits[dataset_name]["train"] : splits[dataset_name]["val"]],
        y[splits[dataset_name]["val"] :],
    )
    logger = logging.getLogger(__name__)
    for n, e in [("x_train", x_train), ("x_val", x_val), ("x_test", x_test)]:
        logger.info(f"Shape of {n}: {e.shape}")

    model, scaler = train_model(
        dataset, model_name, adv_training, x_train, y_train
    )
    test_model(
        model,
        x_test,
        y_test,
    )
    save_model(
        model,
        f"models/{dataset_name}_{model_name}_{'adv' if adv_training else 'clean'}.model",
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

    # Your logic goes here
    run(dataset, model, adv_training, model)
