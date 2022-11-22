import numpy as np
from river.drift import ADWIN


class AdwinErrorDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.drift_detector = None

    def fit(self, **kwarg):
        self.drift_detector = ADWIN(delta=0.002)

    def _update_one(self, y, y_scores):
        error = 1 - y_scores[y]
        in_drift, in_warning = self.drift_detector.update(error)
        return in_drift, in_warning, np.nan

    def update(self, y, y_scores, **kwargs):
        if isinstance(y, np.int64) or (len(y) == 1):
            return self._update_one(y, y_scores)
        elif len(y) >= 2:
            was_drift, was_warning = False, False
            for i in np.arange(len(y)):
                y0, y_scores0 = y[i], y_scores[i]
                in_drift, in_warning, _ = self._update_one(y0, y_scores0)
                was_drift = was_drift or in_drift
                was_warning = was_warning or in_warning
            return was_drift, was_warning, np.nan
        else:
            raise NotImplementedError

    @staticmethod
    def needs_label():
        return True
