import logging
import os
from typing import List

import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset
from mlc.models.model import Model
from mlc.models.model_factory import get_model
from mlc.models.pipeline import Pipeline
from mlc.transformers.tabular_transformer import TabTransformer
from numpy.typing import ArrayLike
from sklearn.base import clone as sk_clone

from drift_study.utils.drift_model import DriftModel
from drift_study.utils.io_utils import load_do_save_model

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def initialize(
    config,
    run_config,
) -> (Dataset, Model, ArrayLike, ArrayLike, ArrayLike):
    logger.debug(f"Loading dataset {config.get('dataset').get('name')}")
    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()

    metadata = dataset.get_metadata(only_x=True)
    model = get_model_arch(
        config, run_config, dataset.get_metadata(only_x=True)
    )
    model = Pipeline(
        steps=[
            TabTransformer(metadata=metadata, scale=True, one_hot_encode=True),
            model,
        ]
    )

    return dataset, model, x, y, t


def get_model_arch(config, run_config, metadata):

    model_class = get_model(run_config.get("model"))
    model = model_class(
        x_metadata=metadata,
        verbose=0,
        n_jobs=config.get("performance").get("n_jobs"),
        random_state=config.get("experience").get("random_state"),
    )

    return model


def get_current_models(models: List[DriftModel], t, last_model_used_i=None):
    # Filter out the model that are known outdated
    idx_to_add = 0
    if last_model_used_i is not None:
        models = models[last_model_used_i:]
        idx_to_add = last_model_used_i
    # From these models, get the one that are in the past
    past_models = list(
        filter(lambda x: x[1].available_time <= t, enumerate(models))
    )
    # Get the latest one, index
    model_i = past_models[-1][0]
    # Re-add the latest model index
    model_i = model_i + idx_to_add
    return model_i


def compute_y_scores(
    model,
    current_index,
    current_model_i,
    model_used,
    y_scores,
    x,
    predict_forward,
):
    if model_used[current_index] < current_model_i:
        logger.debug(f"Seeing forward at index {current_index}")
        x_to_pred = x[current_index : current_index + predict_forward]
        if model.objective in ["regression"]:
            y_pred = model.predict(x_to_pred)
        elif model.objective in ["binary", "classification"]:
            y_pred = model.predict_proba(x_to_pred)
        else:
            raise NotImplementedError
        y_scores[current_index : current_index + predict_forward] = y_pred
        model_used[
            current_index : current_index + predict_forward
        ] = current_model_i
    return y_scores, model_used


def clone_estimator(estimator) -> Pipeline:
    return sk_clone(estimator)


def get_ref_eval_config(configs, ref_config_names):
    ref_configs = []
    eval_configs = []
    for config in configs.get("runs"):
        if config.get("name") in ref_config_names:
            ref_configs.append(config)
        else:
            eval_configs.append(config)
    return ref_configs, eval_configs


def get_common_detectors_params(config, metadata):
    auto_detector_params = {
        "x_metadata": metadata,
        "features": metadata["feature"].to_list(),
        "numerical_features": metadata["feature"][
            metadata["type"] != "cat"
        ].to_list(),
        "categorical_features": metadata["feature"][
            metadata["type"] == "cat"
        ].to_list(),
        "window_size": config.get("window_size"),
    }
    return {**config.get("common_detectors_params"), **auto_detector_params}


def get_delays(run_config, drift_detector):
    delays = run_config.get("delays")
    label_delay = pd.Timedelta(delays.get("label"))
    drift_detection_delay = pd.Timedelta(delays.get("drift"))
    if drift_detector.needs_label():
        drift_detection_delay = drift_detection_delay + label_delay
    retraining_delay = max(label_delay, drift_detection_delay) + pd.Timedelta(
        delays.get("retraining")
    )
    return label_delay, drift_detection_delay, retraining_delay


def add_model(
    models: List[DriftModel],
    model_path,
    model,
    drift_detector,
    t_available,
    x,
    y,
    t,
    start_idx,
    end_idx,
):
    model = load_do_save_model(
        model,
        model_path,
        x.iloc[start_idx:end_idx],
        y[start_idx:end_idx],
    )
    if model.objective in ["regression"]:
        y_scores = model.predict(x.iloc[start_idx:end_idx])
    elif model.objective in ["binary", "classification"]:
        y_scores = model.predict_proba(x.iloc[start_idx:end_idx])
    else:
        raise NotImplementedError
    drift_detector.fit(
        x=x.iloc[start_idx:end_idx],
        t=t[start_idx:end_idx],
        y=y[start_idx:end_idx],
        y_scores=y_scores,
        model=model,
    )

    models.append(DriftModel(t_available, model, drift_detector, 0, end_idx))
