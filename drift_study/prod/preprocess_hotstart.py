import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    data = data.iloc[400000:500000]
    data["unavailable"] = 0.0
    return data
