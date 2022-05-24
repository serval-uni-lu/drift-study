import logging
import os

from evidently import ColumnMapping
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from mlc.datasets import load_datasets

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

configs = [
    {
        "dataset_name": "lcld",
        "model_name": "random_forest",
        "drift_name": "tabular",
        "window_size": 100000,
    }
]


def run(config):
    print(config)
    dataset_name = config.get("dataset_name")
    model_name = config.get("model_name")
    logger.info(f"Starting dataset {dataset_name}, model {model_name}")
    dataset = load_datasets(dataset_name)
    x, y, t = dataset.get_x_y_t()

    iris_data_drift_report = Dashboard(tabs=[DataDriftTab()])

    column_mapping = ColumnMapping()
    column_mapping.numerical_features = dataset.numerical_features
    column_mapping.categorical_features = dataset.categorical_features

    # iris_data_drift_report.calculate(
    #     x[:100000], x[5000:105000], column_mapping=column_mapping
    # )
    iris_data_drift_report.calculate(
        x[0:10000], x[10000:20000], column_mapping=column_mapping
    )
    # iris_data_drift_report.show()
    iris_data_drift_report.save("reports/my_report.html")

    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(
        x[0:10000], x[10000:20000], column_mapping=column_mapping
    )
    report = data_drift_profile.object()
    print(report["data_drift"])


def run_many(configs):
    for config in configs:
        run(config)


if __name__ == "__main__":
    run_many(configs)
