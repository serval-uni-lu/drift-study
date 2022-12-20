from dataclasses import dataclass

import pandas as pd


@dataclass
class Delays:
    label: pd.Timedelta
    drift_detector: pd.Timedelta
    ml_model: pd.Timedelta


def get_delays(run_config: dict, drift_detector) -> Delays:
    delays = run_config.get("delays")
    label_delay = pd.Timedelta(delays.get("label"))
    drift_detection_delay = pd.Timedelta(delays.get("drift"))
    if drift_detector.needs_label():
        drift_detection_delay = drift_detection_delay + label_delay
    retraining_delay = pd.Timedelta(
        max(label_delay, drift_detection_delay)
        + pd.Timedelta(delays.get("retraining"))
    )
    delays = Delays(label_delay, drift_detection_delay, retraining_delay)
    return delays
