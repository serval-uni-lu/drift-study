from typing import List, Union

from drift_study.drift_detectors.river_drift.adwin_drift import AdwinErrorDrift

from .adversarial_drift import AdversarialDrift
from .alibi_drift import detectors as alibi_detectors
from .alibi_drift.tabular_drift import TabularAlibiDrift
from .aries_drift import AriesDrift
from .bayesian_uncertainty.rf_uncertainty_drift import RfUncertaintyDrift
from .evidently_drift import EvidentlyDrift
from .frouros_drift import detectors as frouros_detectors
from .metric_drift import MetricDrift
from .n_batch_drift import NBatchDrift
from .no_drift import NoDrift
from .pca_cd import PcaCdDrift
from .periodic_drift import PeriodicDrift
from .river_drift import detectors as river_detectors

drift_detectors = {
    # "tabular": TabularDrift,
    "periodic": PeriodicDrift,
    "adwin_error": AdwinErrorDrift,
    "evidently": EvidentlyDrift,
    "no_drift": NoDrift,
    "n_batch": NBatchDrift,
    "rf_uncertainty": RfUncertaintyDrift,
    "metric": MetricDrift,
    "pca_cd": PcaCdDrift,
    "aries": AriesDrift,
    "adversarial": AdversarialDrift,
    "tabular_alibi": TabularAlibiDrift,
    **river_detectors,
    **frouros_detectors,
    **alibi_detectors,
}


def get_drift_detectors(drift_detectors_names: Union[str, List[str]]):
    def load_one(drift_detectors_name):
        if drift_detectors_name in drift_detectors:
            return drift_detectors[drift_detectors_name]
        else:
            raise NotImplementedError(
                f"Drift detector '{drift_detectors_name}' is not available."
            )

    if isinstance(drift_detectors_names, str):
        return load_one(drift_detectors_names)
    else:
        return [
            {
                "name": drift_detectors_name,
                "dataset": load_one(drift_detectors_name),
            }
            for drift_detectors_name in drift_detectors_names
        ]
