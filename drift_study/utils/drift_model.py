from dataclasses import dataclass

from mlc.models.model import Model
from numpy.typing import ArrayLike


@dataclass
class DriftModel:
    available_time: ArrayLike
    ml_model: Model
    drift_detector: any
    start_idx: int
    end_idx: int
