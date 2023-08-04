from pathlib import Path
from typing import Union

import pandas as pd

from drift_study.drift_logger.drift_logger import DriftLogger


class ParquetDriftLogger(DriftLogger):
    def __init__(self, path: str):
        self.path = path
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
    ):
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

    def __del__(self):
        if Path(self.path).exists():
            self.df.to_parquet(self.path, engine="fastparquet", append=True)
        else:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self.df.to_parquet(
                self.path,
                engine="fastparquet",
            )

        print("Drift metrics saved to", self.path)
