import numpy as np
import pandas as pd
from river.drift import DDM, EDDM


class DdmDrift:
    def __init__(
        self,
        ddm_args=None,
        **kwargs,
    ) -> None:
        self.drift_detector = None
        self.ddm_args = ddm_args
        if self.ddm_args is None:
            self.ddm_args = {}

    def fit(self, **kwargs):
        self.drift_detector = DDM(**self.ddm_args)

    def _update_one(self, metric, **kwargs):
        in_drift, in_warning = self.drift_detector.update(metric)
        return in_drift, in_warning, pd.DataFrame()

    def update(self, metric, **kwargs):
        if not hasattr(metric, "__len__"):
            return self._update_one(metric)
        else:
            was_drift, was_warning = False, False
            for i in np.arange(len(metric)):
                metric_0 = metric[i]
                in_drift, in_warning, _ = self._update_one(metric_0)
                was_drift = was_drift or in_drift
                was_warning = was_warning or in_warning
            return was_drift, was_warning, pd.DataFrame()


class EddmDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.drift_detector = None

    def fit(self, **kwargs):
        self.drift_detector = EDDM()

    def _update_one(self, metric, **kwargs):
        in_drift, in_warning = self.drift_detector.update(metric)
        return in_drift, in_warning, pd.DataFrame()

    def update(self, metric, **kwargs):
        if not hasattr(metric, "__len__"):
            return self._update_one(metric)
        else:
            was_drift, was_warning = False, False
            for i in np.arange(len(metric)):
                metric_0 = metric[i]
                in_drift, in_warning, _ = self._update_one(metric_0)
                was_drift = was_drift or in_drift
                was_warning = was_warning or in_warning
            return was_drift, was_warning, pd.DataFrame()
