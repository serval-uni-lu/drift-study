from abc import ABC, abstractmethod
from typing import Union


class DriftLogger(ABC):
    @abstractmethod
    def log_metric(
        self,
        metric_name: str,
        metric_value: Union[int, bool, float],
        model_id: str,
        last_transaction_id: str,
        timestamp: int,
    ) -> None:
        pass
