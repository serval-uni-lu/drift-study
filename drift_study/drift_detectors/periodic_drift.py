import numpy as np


class PeriodicDrift:
    def __init__(
        self,
        period,
        **kwargs,
    ) -> None:
        self.period = period
        self.counter = 0

    def fit(self, **kwargs):
        self.counter = 0

    def update(self, x, **kwargs):
        if len(x.shape) == 1:
            self.counter += 1
        elif len(x.shape) == 2:
            self.counter += len(x.shape[0])

        if self.counter < self.period:
            return False, False, np.nan, np.nan
        else:

            return True, True, np.nan, np.nan

    @staticmethod
    def needs_label():
        return False
