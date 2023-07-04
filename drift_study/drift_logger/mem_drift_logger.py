from typing import Union

import pandas as pd

from drift_study.drift_logger.drift_logger import DriftLogger


class MemDriftLogger(DriftLogger):
    def __init__(self) -> None:
        self.df = pd.DataFrame(
            columns=[
                "metric_name",
                "metric_value",
                "model_id",
                "last_transaction_id",
                "timestamp",
            ]
        )

    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, bool, float],
        model_id: str,
        last_transaction_id: str,
        timestamp: int,
    ) -> None:
        new_row = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "model_id": model_id,
            "last_transaction_id": last_transaction_id,
            "timestamp": timestamp,
        }
        self.df = pd.concat(
            [self.df, pd.DataFrame([new_row])], ignore_index=True
        )

    def get_logs(self) -> pd.DataFrame:
        return self.df
