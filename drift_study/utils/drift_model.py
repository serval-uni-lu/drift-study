from dataclasses import dataclass
from typing import List, Tuple

from mlc.models.model import Model
from numpy.typing import ArrayLike

from drift_study.drift_detectors import DriftDetector


@dataclass
class DriftModel:
    ml_available_time: ArrayLike
    ml_model: Model
    drift_available_time: ArrayLike
    drift_detector: DriftDetector
    start_idx: int
    end_idx: int


def get_current_model(
    models: List[DriftModel],
    i,
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

        i_model = model.end_idx
        if (t_model <= t) and (i_model <= i):
            model_idx += 1
        else:
            break

    model_idx = model_idx + idx_to_add
    model_idx = max(0, model_idx)
    return model_idx


def get_current_models(
    models: List[DriftModel],
    i: int,
    t,
    last_ml_model_used=None,
    last_drift_model_used=None,
) -> Tuple[int, int]:
    ml_model_idx, drift_model_idx = (
        get_current_model(models, i, t, "ml", last_ml_model_used),
        get_current_model(models, i, t, "drift", last_drift_model_used),
    )
    if models[drift_model_idx].drift_detector.needs_model():
        assert ml_model_idx == drift_model_idx
    return ml_model_idx, drift_model_idx


def free_mem_models(
    models: List[DriftModel], ml_model_idx: int, drift_model_idx: int
) -> None:
    for i in range(len(models)):
        if i < ml_model_idx:
            models[i].ml_model = None
        if i < drift_model_idx:
            models[i].drift_detector = None
