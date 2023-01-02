import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from evidently import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)

warnings.simplefilter(action="ignore", category=FutureWarning)


class EvidentlyDrift(DriftDetector):
    def __init__(
        self,
        window_size: int,
        numerical_features: List[str] = None,
        categorical_features: List[str] = None,
        threshold: float = 0.05,
        drift_share: float = 0.5,
        fit_first_update: bool = False,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            window_size=window_size,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            threshold=threshold,
            drift_share=drift_share,
            **kwargs,
        )
        self.window_size = window_size
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.drift_share = drift_share

        self.drift_detector: Optional[Profile] = None
        self.column_mapping = ColumnMapping()

        self.x_ref = pd.DataFrame()
        self.x_last = pd.DataFrame()
        self.fit_first_update = fit_first_update
        self.is_first_update = True

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:

        options = [DataDriftOptions(drift_share=self.drift_share)]
        self.drift_detector = Profile(
            sections=[DataDriftProfileSection()], options=options
        )

        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_features
        self.column_mapping.categorical_features = self.categorical_features
        self.x_ref = x
        self.x_last = x
        self.is_first_update = True

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:

        if self.drift_detector is None:
            raise NotFittedDetectorException
        else:
            if self.fit_first_update and self.is_first_update:
                self.window_size = len(x)
                self.fit(x, t, y, y_scores, None)

            self.x_last = pd.concat(
                [
                    self.x_last,
                    pd.DataFrame(x, columns=self.x_last.columns),
                ]
            )[-self.window_size :]

            self.drift_detector.calculate(
                self.x_ref,
                self.x_last,
                column_mapping=self.column_mapping,
            )
            report = self.drift_detector.object()
            in_drift = report["data_drift"]["data"]["metrics"]["dataset_drift"]
            in_warning = (
                report["data_drift"]["data"]["metrics"]["n_drifted_features"]
                > 0
            )

            self.is_first_update = False
            return (
                in_drift,
                in_warning,
                pd.DataFrame(report["data_drift"]["data"]["metrics"].copy()),
            )

    def needs_label(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "threshold": trial.suggest_float("p_val", 1e-4, 0.1),
            "drift_share": trial.suggest_float("drift_share", 1e-6, 1),
        }


detectors: Dict[str, Type[DriftDetector]] = {"evidently": EvidentlyDrift}
