from dataclasses import dataclass

from mlc.models.model import Model
from numpy.typing import ArrayLike


@dataclass
class DriftModel:
    ml_available_time: ArrayLike
    ml_model: Model
    drift_available_time: ArrayLike
    drift_detector: any
    start_idx: int
    end_idx: int
