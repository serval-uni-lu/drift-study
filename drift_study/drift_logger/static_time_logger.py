from typing import Union

from drift_study.drift_logger.drift_logger import DriftLogger


class StaticTimeDriftLogger(DriftLogger):
    def __init__(self, logger: DriftLogger, timestamp: int) -> None:
        self.logger = logger
        self.timestamp = timestamp

    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, bool, float],
        model_id: str,
        last_transaction_id: str,
        timestamp: int,
    ):
        self.logger.log_metric(
            metric_name,
            metric_value,
            model_id,
            last_transaction_id,
            self.timestamp,
        )
