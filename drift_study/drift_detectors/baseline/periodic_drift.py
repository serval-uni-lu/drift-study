from typing import Any, Dict

import optuna


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
            return False, False, None
        else:

            return True, True, None

    @staticmethod
    def needs_label() -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "period": trial.suggest_int(
                "period",
                trial_params["period"]["min"],
                trial_params["period"]["max"],
            )
        }
        return params
