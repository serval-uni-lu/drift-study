from typing import Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from drift_study.drift_logger.drift_logger import DriftLogger


class PostgresDriftLogger(DriftLogger):
    def __init__(self, db_url: str, db_table: str):
        self.db_url = db_url
        self.db_table = db_table
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
        # SQL is typed, what a funny idea
        if isinstance(metric_value, bool) or isinstance(
            metric_value, np.bool_
        ):
            metric_value = int(metric_value)
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
        engine = create_engine(self.db_url)
        self.df.to_sql(self.db_table, engine, if_exists="append")
        print("Drift metrics saved to database")
