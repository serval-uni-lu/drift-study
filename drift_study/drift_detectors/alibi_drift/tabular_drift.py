import numpy as np
import pandas as pd
from alibi_detect.cd import TabularDrift as InternalTabularDrift


class TabularAlibiDrift:
    def __init__(
        self,
        window_size: int,
        p_value: float,
        numerical_features,
        categorical_features,
        x_metadata,
        correction: str,
        **kwargs,
    ) -> None:
        self.window_size = window_size
        self.p_value = p_value
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.x_metadata = x_metadata
        self.correction = correction

        self.x_test = None
        self.drift_detector = None

    def fit(self, x, **kwargs):
        categories_per_feature = {}

        for index, row in self.x_metadata.iterrows():
            if row["type"] == "cat":
                # categories_per_feature[int(index)] = np.arange(
                #     int(row["min"]), int(row["max"]) + 1
                # ).tolist()
                categories_per_feature[int(index)] = None
        self.drift_detector = InternalTabularDrift(
            np.array(x),
            p_val=self.p_value,
            correction=self.correction,
            categories_per_feature=categories_per_feature,
        )
        self.x_test = x

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)
        self.x_test = pd.concat([self.x_test, x])
        self.x_test = self.x_test.iloc[-self.window_size :]

        out = self.drift_detector.predict(
            self.x_test.to_numpy().astype(np.double)
        )
        is_drift = out["data"]["is_drift"] > 0
        is_warning = is_drift
        data = {}
        for e in out["data"].keys():
            data[e] = [out["data"][e]]
        return is_drift, is_warning, pd.DataFrame.from_dict(data)

    @staticmethod
    def needs_label():
        return False
