import abc
import logging
from abc import ABC
from typing import Any, Dict, Optional, Tuple, Union

import optuna
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_cache.drift_cache import DriftCache
from drift_study.drift_logger.drift_logger import DriftLogger
from drift_study.drift_logger.empty_drift_logger import EmptyDriftLogger
from drift_study.typing import NDFloat, NDInt, NDNumber


class NotFittedDetectorException(Exception):
    """Raised when the DriftDetector is not fitted but update is called."""

    pass


class NoModelException(Exception):
    """Raised when the DriftDetector needs the model but does not find one."""

    pass


class DriftDetector(ABC):
    def __init__(
        self,
        drift_logger: Optional[DriftLogger] = None,
        fit_cache: Optional[DriftCache] = None,
        **kwargs: Any,
    ) -> None:
        self.fit_cache = fit_cache
        if drift_logger is None:
            self.drift_logger = EmptyDriftLogger()
        else:
            self.drift_logger = drift_logger
        self.kwargs = kwargs

    @abc.abstractmethod
    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
        model: Optional[Model],
    ) -> None:
        pass

    @abc.abstractmethod
    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
    ) -> Tuple[bool, bool, pd.DataFrame]:
        pass

    @staticmethod
    @abc.abstractmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        pass

    @abc.abstractmethod
    def needs_model(self) -> bool:
        pass

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        logger = logging.getLogger(__name__)
        logger.warning("Default parameters not set.")
        return None
