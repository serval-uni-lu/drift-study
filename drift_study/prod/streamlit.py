import pandas as pd
import streamlit as st
from sqlalchemy import create_engine


def show_one_metric(df: pd.DataFrame, metric_name: str) -> None:
    df = df[df["metric_name"] == metric_name]
    st.write(f"## {metric_name}")
    st.line_chart(df[["timestamp", "metric_value"]].set_index("timestamp"))


def run() -> None:
    USERNAME = "postgres"
    PASSWORD = "example"
    DATABASE = "lcld"
    SERVER = "localhost"

    METRIC_TO_PRINT = ["jsd_drift_share"]

    engine = create_engine(
        f"postgresql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    )

    query = (
        "SELECT metric_name, metric_value, timestamp "
        "FROM drift_metrics WHERE timestamp <= 1640995200"
    )

    df = pd.read_sql(query, engine)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    for metric_name in METRIC_TO_PRINT:
        show_one_metric(df, metric_name)

    # print(df)


if __name__ == "__main__":
    run()
