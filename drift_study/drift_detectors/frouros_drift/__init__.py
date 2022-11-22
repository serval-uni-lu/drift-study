from typing import Type, Union

import pandas as pd
from frouros.unsupervised.base import UnsupervisedBaseEstimator
from frouros.unsupervised.statistical_test import KSTest


class UnsupervisedFrourosDrift:
    def __init__(
        self,
        window_size: int,
        internal_detector_cls: Type[UnsupervisedBaseEstimator],
        internal_args: Union[dict, None] = None,
        alpha: float = 0.01,
        **kwargs,
    ) -> None:
        self.internal_detector_cls = internal_detector_cls

        self.internal_args = internal_args
        if self.internal_args is None:
            self.internal_args = {}
        self.drift_detector = None
        self.window_size = window_size
        self.x_test = None

    def fit(self, x, **kwargs):
        self.drift_detector = self.internal_detector_cls()
        self.drift_detector.fit(x)
        self.x_test = x

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)
        self.x_test = pd.concat([self.x_test, x])
        self.x_test = self.x_test.iloc[-self.window_size :]
        return False, False, pd.DataFrame()

    @staticmethod
    def needs_label():
        return False


class KSTestDrift(UnsupervisedFrourosDrift):
    def __init__(
        self,
        window_size: int,
        internal_args: Union[dict, None] = None,
        **kwargs,
    ):
        super().__init__(window_size, KSTest, internal_args, **kwargs)


detectors = {
    "ks_test": KSTestDrift,
}
