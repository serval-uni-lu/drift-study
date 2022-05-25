import numpy as np


class NoDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        pass

    def fit(self, x, t, y, y_scores):
        pass

    def update(self, x, t, y, y_scores):
        return False, False, np.nan, np.nan
