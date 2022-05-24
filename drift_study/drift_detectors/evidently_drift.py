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
        batch,
        numerical_features,
        categorical_features,
        **kwargs,
    ) -> None:
        self.window_size = window_size
        self.batch = batch
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.drift_detector = None
        self.column_mapping = None
        self.x_ref = None
        self.nb_mem = 0
        self.x_mem = []
        self.x_last_used = None

    def fit(self, x, y, y_scores):
        options = [DataDriftOptions(drift_share=0.01)]
        self.drift_detector = Profile(
            sections=[DataDriftProfileSection()], options=options
        )

        self.column_mapping = ColumnMapping()
        self.column_mapping.numerical_features = self.numerical_features
        self.column_mapping.categorical_features = self.categorical_features
        self.x_ref = x
        self.nb_mem = 0
        self.x_mem = []
        self.x_last_used = x

    def update(self, x, y, y_scores):
        self.x_mem.append(x)
        self.nb_mem += 1
        if self.nb_mem < self.batch:
            return False, False, np.nan, np.nan
        else:
            self.x_last_used = pd.concat(
                [
                    self.x_last_used,
                    pd.DataFrame(self.x_mem, columns=self.x_last_used.columns),
                ]
            )[-self.window_size :]

            self.drift_detector.calculate(
                self.x_ref,
                self.x_last_used,
                column_mapping=self.column_mapping,
            )
            report = self.drift_detector.object()

            self.x_mem = []
            self.nb_mem = 0

            in_drift = report["data_drift"]["data"]["metrics"]["dataset_drift"]
            in_warning = (
                report["data_drift"]["data"]["metrics"]["n_drifted_features"]
                > 0
            )

            return in_drift, in_warning, np.nan, np.nan
