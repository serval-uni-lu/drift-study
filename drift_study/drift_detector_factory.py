from typing import List, Union

from drift_study.drift_detectors import get_drift_detectors


def get_drift_detector(drift_detectors_names: Union[str, List[str]], **kwargs):

    if isinstance(drift_detectors_names, str):
        return get_drift_detectors(drift_detectors_names)(**kwargs)
    else:
        drift_detectors_names.reverse()
        detector = None
        for element in drift_detectors_names:
            local_detector_class = get_drift_detectors(element)
            detector = local_detector_class(**kwargs, drift_detector=detector)

        return detector
