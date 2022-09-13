import numpy as np


class NoDrift:
    def __init__(
        self,
        **kwargs,
    ) -> None:
        # No drift does not require parameters
        # Used to be compliant with others drift detectors
        pass

    def fit(self, **kwargs):
        # No drift does not require fitting
        # Used to be compliant with others drift detectors
        pass

    def update(self, **kwargs):
        return False, False, np.nan, np.nan

    @staticmethod
    def needs_label():
        return False
