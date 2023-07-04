from typing import Tuple

import numpy as np
import pandas as pd
from mlc.datasets.samples import load_dataset

from drift_study.drift_detectors.data_based.divergence_drift import (
    DivergenceDrift,
)
from drift_study.drift_logger.mem_drift_logger import MemDriftLogger


class TestDivergenceDrift:
    @staticmethod
    def get_known_distributions() -> Tuple[pd.DataFrame, pd.DataFrame]:

        ref_values = np.arange(1, 5)
        ref_bin = [1, 2, 3, 4]
        ref_dist = np.concatenate(
            [[ref_values[i]] * ref_bin[i] for i in range(len(ref_values))]
        )

        reference_distribution = pd.DataFrame(ref_dist, columns=["a"])

        test_bin = [13, 14, 15, 16]
        test_bin = np.concatenate(
            [[ref_values[i]] * test_bin[i] for i in range(len(ref_values))]
        )
        test_distribution = pd.DataFrame(test_bin, columns=["a"])

        return reference_distribution, test_distribution

    def test_observation(self) -> None:

        drift_logger = MemDriftLogger()
        drift_detector = DivergenceDrift(
            numerical_features=["a"],
            categorical_features=[],
            n_bin=4,
            drift_logger=drift_logger,
        )

        (
            reference_distribution,
            test_distribution,
        ) = self.get_known_distributions()
        drift_detector.fit(reference_distribution, None, None, None, None)

        drift_detector.evaluate(test_distribution, None, None, None)

        logs = drift_logger.get_logs()
        metric = logs[logs["metric_name"] == "jsd_num_a"]["metric_value"].iloc[
            0
        ]

        assert np.isclose(metric, 0.1402339).all()

    def test_no_drift_logger_no_crash(self) -> None:

        drift_detector = DivergenceDrift(
            numerical_features=["a"],
            categorical_features=[],
            n_bin=4,
        )

        (
            reference_distribution,
            test_distribution,
        ) = self.get_known_distributions()
        drift_detector.fit(reference_distribution, None, None, None, None)

        drift_detector.evaluate(test_distribution, None, None, None)

    def test_double_observation(self) -> None:
        drift_logger = MemDriftLogger()
        drift_detector = DivergenceDrift(
            numerical_features=["a"],
            categorical_features=[],
            n_bin=4,
            drift_logger=drift_logger,
        )

        (
            reference_distribution,
            test_distribution,
        ) = self.get_known_distributions()

        drift_detector.fit(reference_distribution, None, None, None, None)

        drift_detector.evaluate(test_distribution, None, None, None)
        drift_detector.evaluate(
            pd.concat([test_distribution, test_distribution]), None, None, None
        )

        logs = drift_logger.get_logs()
        metrics = logs[logs["metric_name"] == "jsd_num_a"]["metric_value"]

        assert metrics.iloc[0] == metrics.iloc[1]

    def test_double_observation_real_data(self) -> None:

        dataset = load_dataset("lcld_201317_ds_time")
        x, _ = dataset.get_x_y()
        x_metata = dataset.get_metadata(only_x=True)

        params = {
            "numerical_features": x_metata["feature"][
                x_metata["type"] != "cat"
            ].to_list(),
            "categorical_features": x_metata["feature"][
                x_metata["type"] == "cat"
            ].to_list(),
        }

        drift_logger = MemDriftLogger()
        drift_detector = DivergenceDrift(
            **params,
            drift_logger=drift_logger,
        )

        reference_distribution = x.iloc[:400000]
        test_distribution = x.iloc[400000:800000]

        drift_detector.fit(reference_distribution, None, None, None, None)
        drift_detector.evaluate(test_distribution, None, None, None)
        drift_detector.evaluate(
            pd.concat([test_distribution, test_distribution]), None, None, None
        )
        logs = drift_logger.get_logs()

        metrics_to_check = [
            e for e in logs["metric_name"].unique() if "jsd" in e
        ]
        for metric in metrics_to_check:
            metrics = logs[logs["metric_name"] == metric]["metric_value"]
            assert metrics.iloc[0] == metrics.iloc[1]
