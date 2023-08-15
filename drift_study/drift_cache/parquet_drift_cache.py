from os import listdir
from pathlib import Path
from typing import Dict

import pandas as pd

from drift_study.drift_cache.drift_cache import DriftCache


class ParquetDriftCache(DriftCache):
    def __init__(self, path: str, compression="gzip") -> None:
        self.path = path
        self.compression = compression
        self.cache_data
        super().__init__()

    def save(self, caches: Dict[str, pd.DataFrame]) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.cache_data = caches
        for c in caches:
            caches[c].to_parquet(
                f"{self.path}/{c}.parquet", compression=self.compression
            )

    def load(self) -> Dict[str, pd.DataFrame]:
        files = [e for e in listdir(self.path) if Path(e).suffix == ".parquet"]
        out = {}
        for e in files:
            out[Path(e).stem] = pd.read_parquet(f"{self.path}/{e}")
        self.cache_data = out
        return out

    def _load_key(self, name: str) -> pd.DataFrame:
        df = pd.read_parquet(f"{self.path}/{name}.parquet")
        return df
    
    def get(self, name: str) -> pd.DataFrame:
        if self.cache_data is None:
            self.cache_data = {
                name: self._load_key(name)
            }
        if name not in self.cache_data:
            self.cache_data[name] = self._load_key(name)
        return self.cache_data[name]
