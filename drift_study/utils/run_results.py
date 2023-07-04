import dataclasses

from drift_study.typing import NDBool, NDInt, NDNumber


@dataclasses
class RunResult:

    model_used: NDInt
    model_start_idxs: NDInt
    model_end_idxs: NDInt

    y_scores: NDNumber
    ml_metric: float

    is_drifts: NDBool
    is_drift_warnings: NDBool

    @property
    def n_train(self) -> int:
        return int(self.model_used.max() + 1)
