import warnings

import pandas

from drift_study.drift_detectors.bayesian_uncertainty.rf_uncertainty import (
    RandomForestClassifierWithUncertainty,
)

warnings.simplefilter(action="ignore", category=FutureWarning)

UNCERTAINTY_TYPE = {"total": 0, "epistemic": 1, "aleatoric": 2}


class RfUncertaintyDrift:
    def __init__(
        self,
        drift_detector,
        uncertainty_type: str,
        **kwargs,
    ) -> None:
        self.drift_detector = drift_detector
        self.uncertainty_type = uncertainty_type
        self.rf_uncertainty = None
        self.model = None
        self.x_last = None

    def fit(self, x, y, model, **kwargs):
        self.model = model
        self.rf_uncertainty = RandomForestClassifierWithUncertainty(
            random_forest=model[-1]
        )
        self.rf_uncertainty.fit(model[:-1].transform(x), y)
        self.drift_detector.fit(x=x, y=y, model=model, **kwargs)

    def update(self, x, **kwargs):
        x = pandas.DataFrame(x)
        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model[:-1].transform(x)
        )

        return self.drift_detector.update(
            [e[UNCERTAINTY_TYPE[self.uncertainty_type]] for e in uncertainties]
        )

    @staticmethod
    def needs_label():
        return False
