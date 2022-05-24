import numpy as np
from river.drift import ADWIN


class AdwinDrift:
    def __init__(
        self,
        batch_size=None,
        **kwargs,
    ) -> None:
        self.drift_detector = None
        self.batch_size = batch_size
        self.counter = 0
        self.was_drift = False

    def fit(self, x, y, y_scores):
        self.drift_detector = ADWIN(delta=0.002)
        self.counter = 0
        self.was_drift = False

    def update(self, x, y, y_scores):
        error = 1 - y_scores[y]
        in_drift, in_warning = self.drift_detector.update(error)
        if self.batch_size is not None:
            self.counter += 1

            if in_drift:
                self.was_drift = True

            if (self.counter % self.batch_size == 0) and self.was_drift:
                return True, in_warning, np.nan, np.nan
            else:
                return False, in_warning, np.nan, np.nan

        return in_drift, in_warning, np.nan, np.nan
