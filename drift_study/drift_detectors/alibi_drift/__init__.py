from typing import Type, Union

import numpy as np
import pandas as pd
from alibi_detect.base import DriftConfigMixin
from alibi_detect.cd import LearnedKernelDrift, LSDDDrift, MMDDrift
from alibi_detect.cd import SpotTheDiffDrift as AlibiSpotTheDiffDrift
from alibi_detect.utils.pytorch import GaussianRBF


class AlibiDrift:
    def __init__(
        self,
        window_size: int,
        internal_detector_cls: Type[DriftConfigMixin],
        internal_args: Union[dict, None] = None,
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
        self.drift_detector = self.internal_detector_cls(
            x_ref=x.to_numpy().astype(np.double),
            backend="pytorch",
            **self.internal_args,
        )
        self.x_test = x

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)
        self.x_test = pd.concat([self.x_test, x])
        self.x_test = self.x_test.iloc[-self.window_size :]

        out = self.drift_detector.predict(
            self.x_test.to_numpy().astype(np.double)
        )
        is_drift = out["data"]["is_drift"] > 0
        is_warning = is_drift
        data = {}
        for e in out["data"].keys():
            data[e] = [out["data"][e]]
        return is_drift, is_warning, pd.DataFrame.from_dict(data)

    @staticmethod
    def needs_label():
        return False


class MmdDrift(AlibiDrift):
    def __init__(self, window_size, **kwargs):
        super().__init__(window_size, MMDDrift, **kwargs)


class LsddDrift(AlibiDrift):
    def __init__(self, window_size: int, **kwargs):
        super().__init__(window_size, LSDDDrift, **kwargs)


class MmdLkDrift(AlibiDrift):
    def __init__(self, window_size: int, **kwargs):
        add_args = {"kernel": GaussianRBF()}
        if "internal_args" in kwargs:
            kwargs["internal_args"] = (
                {**kwargs["internal_args"] ** add_args},
            )
        else:
            kwargs["internal_args"] = add_args
        super().__init__(window_size, LearnedKernelDrift, **kwargs)


class SpotTheDiffDrift(AlibiDrift):
    def __init__(self, window_size: int, **kwargs):
        super().__init__(window_size, AlibiSpotTheDiffDrift, **kwargs)


detectors = {
    "mmd": MmdDrift,
    "mmd_lk": MmdLkDrift,
}
