from typing import List, Union

from .adwin_drift import AdwinDrift, AdwinErrorDrift
from .bayesian_uncertainty.rf_uncertainty_drift import RfUncertaintyDrift
from .ddm_drift import DdmDrift, EddmDrift
from .evidently_drift import EvidentlyDrift
from .metric_drift import MetricDrift
from .n_batch_drift import NBatchDrift
from .no_drift import NoDrift
from .pca_cd import PcaCdDrift
from .periodic_drift import PeriodicDrift
from .tabular_drift import TabularDrift

drift_detectors = {
    "tabular": TabularDrift,
    "periodic": PeriodicDrift,
    "adwin": AdwinDrift,
    "adwin_error": AdwinErrorDrift,
    "evidently": EvidentlyDrift,
    "no_drift": NoDrift,
    "n_batch": NBatchDrift,
    "rf_uncertainty": RfUncertaintyDrift,
    "metric": MetricDrift,
    "ddm": DdmDrift,
    "eddm": EddmDrift,
    "pca_cd": PcaCdDrift,
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
