import numpy as np
from river.drift import ADWIN


class AdwinDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.drift_detector = None

    def fit(self, x, y, y_scores):
        self.drift_detector = ADWIN(delta=0.002)

    def _update_one(self, x, t, y, y_scores):
        error = 1 - y_scores[y]
        in_drift, in_warning = self.drift_detector.update(error)
        return in_drift, in_warning, np.nan, np.nan

    def update(self, x, t, y, y_scores):
        if len(x.shape) == 1:
            return self._update_one(x, t, y, y_scores)
        elif len(x.shape) == 2:
            was_drift, was_warning = False, False
            for x0, t0, y0, y_scores0 in zip(x, t, y, y_scores):

                in_drift, in_warning, _, _ = self._update_one(
                    x0, t0, y0, y_scores0
                )
                was_warning = was_warning or in_drift
                was_warning = was_warning or was_warning
            return was_drift, was_warning, np.nan, np.nan
        else:
            raise NotImplementedError
