from typing import List, Union

from drift_study.drift_logger.drift_logger import DriftLogger


class MultiDriftLogger(DriftLogger):
    def __init__(self, loggers: List[DriftLogger]) -> None:
        self.loggers = loggers

    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, bool, float],
        model_id: str,
        last_transaction_id: str,
        timestamp: int,
    ):
        for logger in self.loggers:
            logger.log_metric(
                metric_name,
                metric_value,
                model_id,
                last_transaction_id,
                timestamp,
            )
