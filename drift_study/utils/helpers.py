import logging
from multiprocessing import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.datasets.dataset_factory import get_dataset
from mlc.models.model import Model
from numpy.typing import ArrayLike

from drift_study.drift_detectors import DriftDetector
from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.utils.datasets import update_dataset_name
from drift_study.utils.date_sampler import sample_date
from drift_study.utils.delays import Delays
from drift_study.utils.detector import get_f_new_detector
from drift_study.utils.drift_model import DriftModel
from drift_study.utils.io_utils import load_do_save_model
from drift_study.utils.model import get_f_new_model, quite_model

# def batch_size_fix(config: Dict[str, Any], schedule_config: Dict[str, Any]):
#     batch_size = 
#     detectors = schedule_config["detectors"]
    
#     for i in range(len(schedule_config["detectors"])):
#         e = detectors[i]
#         if e["name"] == "n_batch":
#             params = e.get("params")
#             if params if None:
#                 e["params"] = {"batch_size": con}

def initialize(
    config: Dict[str, Any],
) -> Tuple[
    Dataset,
    Callable[[], Model],
    Callable[[int, int], DriftDetector],
    ArrayLike,
    ArrayLike,
    ArrayLike,
]:
    logger = logging.getLogger(__name__)
    logger.debug(f"Loading dataset {config.get('dataset', {}).get('name')}")
    dataset = get_dataset(config.get("dataset"))
    x, y, t = dataset.get_x_y_t()
    x, y, t = sample_date(x, y, t, config.get("sampling_minority_share"))
    update_dataset_name(dataset, config.get("sampling_minority_share"))

    metadata = dataset.get_metadata(only_x=True)
    f_new_model = get_f_new_model(config.get("model"), metadata)
    f_new_detector = get_f_new_detector(
        config["schedule"]["detectors"], metadata, config.get("common_detectors_params", {})
    )
    return dataset, f_new_model, f_new_detector, x, y, t


def compute_y_scores(
    model: Model,
    current_index: int,
    current_model_i: int,
    model_used: np.ndarray,
    y_scores: np.ndarray,
    x: Union[np.ndarray, pd.DataFrame],
    predict_forward: int,
    last_idx: int,
):
    logger = logging.getLogger(__name__)
    if model_used[current_index] < current_model_i:
        logger.debug(f"Seeing forward at index {current_index}")
        end_idx = min(current_index + predict_forward, last_idx)

        if isinstance(model, LazyPipeline):
            y_scores[current_index:end_idx] = model.lazy_predict(
                current_index, end_idx
            )
        else:
            x_to_pred = x[current_index:end_idx]
            if model.objective in ["regression"]:
                y_pred = model.predict(x_to_pred)
            elif model.objective in ["binary", "classification"]:
                y_pred = model.predict_proba(x_to_pred)
            else:
                raise NotImplementedError
            y_scores[current_index:end_idx] = y_pred

        model_used[current_index:end_idx] = current_model_i
    return y_scores, model_used


def get_ref_eval_config(configs: dict, ref_config_names: List[str]):
    ref_configs = []
    eval_configs = []
    for config in configs.get("runs"):
        if config.get("name") in ref_config_names:
            ref_configs.append(config)
        else:
            eval_configs.append(config)
    return ref_configs, eval_configs


def add_model(
    models: List[DriftModel],
    model_path: str,
    f_new_model: Callable[[], Model],
    f_new_detector: Callable[[int, int], DriftDetector],
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
    model = f_new_model()
    model = load_do_save_model(
        model,
        model_path,
        x[start_idx:end_idx],
        y[start_idx:end_idx],
        lock_model_writing,
        list_model_writing,
    )
    quite_model(model)

    if isinstance(model, LazyPipeline):
        y_scores = model.safe_lazy_predict(x, start_idx, end_idx)
        model.safe_lazy_many_predict(x, 50, start_idx, end_idx)
    else:
        if model.objective in ["regression"]:
            y_scores = model.predict(x[start_idx:end_idx])
        elif model.objective in ["binary", "classification"]:
            y_scores = model.predict_proba(x[start_idx:end_idx])
        else:
            raise NotImplementedError

    drift_detector = f_new_detector(start_idx, end_idx)
    drift_detector.fit(
        x=x[start_idx:end_idx],
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
