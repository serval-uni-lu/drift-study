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
        self.rf = None
        self.model = None
        self.x_last = None

    def fit(self, x, y, model, **kwargs):
        internal_model = model[-1]
        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
        self.rf = internal_model
        self.model = model

        self.rf_uncertainty = RandomForestClassifierWithUncertainty(
            random_forest=self.rf
        )
        self.rf_uncertainty.fit(self.model.transform(x), y)
        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model.transform(x)
        )
        metric = [
            e[UNCERTAINTY_TYPE[self.uncertainty_type]] for e in uncertainties
        ]
        self.drift_detector.fit(x=x, y=y, model=model, metric=metric, **kwargs)

    def update(self, x, **kwargs):
        x = pandas.DataFrame(x)
        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model.transform(x)
        )

        uncertainties = [
            e[UNCERTAINTY_TYPE[self.uncertainty_type]] for e in uncertainties
        ]

        is_drift, is_warning, metrics = self.drift_detector.update(
            metric=uncertainties
        )
        metrics[f"{self.uncertainty_type}_uncertainty"] = uncertainties
        return is_drift, is_warning, metrics

    @staticmethod
    def needs_label():
        return False
