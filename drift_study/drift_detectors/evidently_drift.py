import warnings

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.options import DataDriftOptions

warnings.simplefilter(action="ignore", category=FutureWarning)


class EvidentlyDrift:
    def __init__(
        self,
        window_size,
        numerical_features=None,
        categorical_features=None,
        **kwargs,
    ) -> None:
        self.window_size = window_size
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.drift_detector = None
        self.column_mapping = None

        self.x_ref = None
        self.x_last = None

    def fit(self, **kwargs):
        if "metric" in kwargs:
            x = pd.DataFrame(kwargs["metric"], columns=["metric"])
            self.numerical_features = ["metric"]
            self.categorical_features = None
        else:
            x = kwargs["x"]
        options = [DataDriftOptions(drift_share=0.01)]
        self.drift_detector = Profile(
            sections=[DataDriftProfileSection()], options=options
        )

        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_features
        self.column_mapping.categorical_features = self.categorical_features
        self.x_ref = x
        self.x_last = x

    def update(self, **kwargs):
        if "metric" in kwargs:
            x = pd.DataFrame(kwargs["metric"], columns=["metric"])
        else:
            x = kwargs["x"]

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
            report["data_drift"]["data"]["metrics"]["n_drifted_features"] > 0
        )

        return in_drift, in_warning, np.nan, np.nan

    @staticmethod
    def needs_label():
        return False
