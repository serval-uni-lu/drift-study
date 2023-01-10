import logging
import os
from multiprocessing import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset
from mlc.models.model import Model
from mlc.models.model_factory import get_model
from mlc.models.pipeline import Pipeline
from mlc.models.sk_models import SkModel
from mlc.transformers.tabular_transformer import TabTransformer
from numpy.typing import ArrayLike
from sklearn.base import clone as sk_clone

from drift_study.utils.delays import Delays
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.io_utils import load_do_save_model

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)


def initialize(
    config: Dict[str, Any],
    run_config: Dict[str, Any],
) -> Tuple[Dataset, Model, ArrayLike, ArrayLike, ArrayLike]:
    logger.debug(f"Loading dataset {config.get('dataset', {}).get('name')}")
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


def get_model_arch(
    config: Dict[str, Any], run_config: Dict[str, Any], metadata: pd.DataFrame
) -> Model:

    model_class = get_model(run_config.get("model"))
    model = model_class(
        x_metadata=metadata,
        verbose=0,
        n_jobs=config.get("performance", {"n_jobs": 1}).get("n_jobs", 1),
        random_state=config.get("experience", {}).get("random_state"),
    )

    return model


def get_current_model(
    models: List[DriftModel],
    t,
    model_type: str,
    last_model_used_idx=None,
):
    idx_to_add = 0
    if last_model_used_idx is not None:
        models = models[last_model_used_idx:]
        idx_to_add = last_model_used_idx

    model_idx = -1
    for model in models:
        if model_type == "ml":
            t_model = model.ml_available_time
        elif model_type == "drift":
            t_model = model.drift_available_time
        else:
            raise NotImplementedError

        if t_model <= t:
            model_idx += 1
        else:
            break

    model_idx = model_idx + idx_to_add
    model_idx = max(0, model_idx)
    return model_idx


def get_current_models(
    models: List[DriftModel],
    t,
    last_ml_model_used=None,
    last_drift_model_used=None,
) -> Tuple[int, int]:

    return (
        get_current_model(models, t, "ml", last_ml_model_used),
        get_current_model(models, t, "drift", last_drift_model_used),
    )


def compute_y_scores(
    model: Model,
    current_index: int,
    current_model_i: int,
    model_used: np.ndarray,
    y_scores: np.ndarray,
    x: Union[np.ndarray, pd.DataFrame],
    predict_forward: int,
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


def get_ref_eval_config(configs: dict, ref_config_names: List[str]):
    ref_configs = []
    eval_configs = []
    for config in configs.get("runs"):
        if config.get("name") in ref_config_names:
            ref_configs.append(config)
        else:
            eval_configs.append(config)
    return ref_configs, eval_configs


def get_common_detectors_params(config: dict, metadata: pd.DataFrame):
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


def quite_model(model: Model):
    l_model = model
    if isinstance(l_model, Pipeline):
        l_model = l_model[-1]
    if isinstance(l_model, SkModel):
        l_model.model.set_params(**{"verbose": 0})


def add_model(
    models: List[DriftModel],
    model_path: str,
    model,
    drift_detector,
    x_idx,
    delays: Delays,
    x,
    y,
    t,
    start_idx,
    end_idx,
    lock_model_writing: Optional[Lock] = None,
    list_model_writing: Optional[Dict[str, Any]] = None,
) -> None:
    model = load_do_save_model(
        model,
        model_path,
        x.iloc[start_idx:end_idx],
        y[start_idx:end_idx],
        lock_model_writing,
        list_model_writing,
    )
    quite_model(model)
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

    models.append(
        DriftModel(
            t[x_idx] + delays.ml_model,
            model,
            t[x_idx] + delays.drift_detector,
            drift_detector,
            0,
            end_idx,
        )
    )
