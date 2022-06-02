from typing import List, Union

from .adwin_drift import AdwinDrift
from .evidently_drift import EvidentlyDrift
from .n_batch_drift import NBatchDrift
from .no_drift import NoDrift
from .periodic_drift import PeriodicDrift
from .tabular_drift import TabularDrift

drift_detectors = {
    "tabular": TabularDrift,
    "periodic": PeriodicDrift,
    "adwin": AdwinDrift,
    "evidently": EvidentlyDrift,
    "no_drift": NoDrift,
    "n_batch": NBatchDrift,
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
