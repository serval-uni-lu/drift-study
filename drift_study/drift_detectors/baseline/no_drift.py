from typing import Any, Dict


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
        return False, False, None

    @staticmethod
    def needs_label() -> bool:
        return False

    @staticmethod
    def define_trial_parameters() -> Dict[str, Any]:
        params = {}
        return params
