from typing import Union

from drift_study.drift_logger.drift_logger import DriftLogger


class EmptyDriftLogger(DriftLogger):
    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, bool, float],
        model_id: str,
        last_transaction_id: str,
        timestamp: int,
    ) -> None:
        # Empty implementation of a drift logger.
        pass
