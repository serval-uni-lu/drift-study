from os import listdir
from pathlib import Path
from typing import Dict

import pandas as pd

from drift_study.drift_cache.drift_cache import DriftCache


class ParquetDriftCache(DriftCache):
    def __init__(self, path: str, compression="gzip") -> None:
        self.path = path
        self.compression = compression
        super().__init__()

    def save(self, caches: Dict[str, pd.DataFrame]) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)
        for c in caches:
            caches[c].to_parquet(
                f"{self.path}/{c}.parquet", compression=self.compression
            )

    def load(self) -> Dict[str, pd.DataFrame]:
        files = [e for e in listdir(self.path) if Path(e).suffix == ".parquet"]
        out = {}
        for e in files:
            out[Path(e).stem] = pd.read_parquet(f"{self.path}/{e}")
        return out
