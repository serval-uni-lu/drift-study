from typing import Type, Union

import numpy as np
import pandas as pd
from river import drift
from river.base import DriftDetector


class RiverDrift:
    def __init__(
        self,
        internal_detector_cls: Type[DriftDetector],
        internal_args: Union[dict, None] = None,
        **kwargs,
    ) -> None:
        self.internal_detector_cls = internal_detector_cls

        self.internal_args = internal_args
        if self.internal_args is None:
            self.internal_args = {}
        self.drift_detector = None

    def fit(self, **kwargs):
        self.drift_detector = self.internal_detector_cls(**self.internal_args)

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

    @staticmethod
    def needs_label():
        return True


class AdwinDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.ADWIN, internal_args, **kwargs)


class DdmDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.DDM, internal_args, **kwargs)


class EddmDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.EDDM, internal_args, **kwargs)


class HdddmADrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.HDDM_A, internal_args, **kwargs)


class HdddmWDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.HDDM_W, internal_args, **kwargs)


class KswinDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.KSWIN, internal_args, **kwargs)


class PageHinkleyDrift(RiverDrift):
    def __init__(self, internal_args: Union[dict, None] = None, **kwargs):
        super().__init__(drift.PageHinkley, internal_args, **kwargs)


detectors = {
    "adwin": AdwinDrift,
    "ddm": DdmDrift,
    "eddm": EddmDrift,
    "hddm_a": HdddmADrift,
    "hddm_w": HdddmWDrift,
    "kswin": KswinDrift,
    "page_hinkley": PageHinkleyDrift,
}
