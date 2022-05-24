import numpy as np


class PeriodicDrift:
    def __init__(
        self,
        period,
        **kwargs,
    ) -> None:
        self.period = period
        self.counter = 0

    def fit(self, x, y, y_scores):
        self.counter = 0

    def update(self, x, y, y_scores):
        self.counter += 1
        if self.counter < self.period:
            return False, False, np.nan, np.nan
        else:

            return True, True, np.nan, np.nan
