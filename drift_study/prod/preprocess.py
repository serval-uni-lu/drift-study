import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.sample(n=1000)
    data["unavailable"] = 0.0
    return data.sample(n=1000)
