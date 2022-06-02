import numpy as np


class NoDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # No drift does not require parameters
        # Used to be compliant with others drift detectors
        pass

    def fit(self, x, t, y, y_scores):
        # No drift does not require fitting
        # Used to be compliant with others drift detectors
        pass

    def update(self, x, t, y, y_scores):
        return False, False, np.nan, np.nan

    @staticmethod
    def needs_label():
        return False
