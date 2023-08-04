from abc import ABC
from typing import Dict

import pandas as pd


class DriftCache(ABC):
    def save(self, caches: Dict[str, pd.DataFrame]) -> None:
        raise NotImplementedError()

    def load(self) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError()
