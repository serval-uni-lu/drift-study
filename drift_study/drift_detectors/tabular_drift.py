import numpy as np
from alibi_detect.cd import TabularDrift as InternalTabularDrift


class TabularDrift:
    def __init__(
        self,
        window_size,
        batch,
        p_val,
        features,
        numerical_features,
        categorical_features,
        **kwargs,
    ) -> None:
        self.window_size = window_size
        self.batch = batch
        self.p_val = p_val
        self.features = features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.drift_detector = None
        self.x_last_used = None
        self.count_drift = 0
        self.nb_mem = 0
        self.x_mem = []

    def fit(self, x, y, y_scores):
        categories_per_feature = {}
        for f in self.categorical_features:
            categories_per_feature[
                np.argwhere(self.features == f)[0][0]
            ] = None
        self.drift_detector = InternalTabularDrift(
            np.array(x),
            p_val=self.p_val,
            categories_per_feature=categories_per_feature,
        )
        self.nb_mem = 0
        self.x_mem = []
        self.x_last_used = x
        self.count_drift = 0

    def update(self, x, y, y_scores):
        self.x_mem.append(x)
        self.nb_mem += 1
        if self.nb_mem < self.batch:
            return False, False, np.nan, np.nan
        else:
            self.x_last_used = np.concatenate(
                [np.array(self.x_last_used), self.x_mem], axis=0
            )[-self.window_size :]

            preds = self.drift_detector.predict(
                self.x_last_used,
                drift_type="feature",
                return_p_val=True,
                return_distance=True,
            )
            self.x_mem = []
            self.nb_mem = 0
            if preds["data"]["is_drift"].max():
                self.count_drift = self.count_drift + 1
                # for i in np.argwhere(preds["data"]["is_drift"] > 0):
                #     print(f"Warning detected on {self.features[i]}")
                if self.count_drift > 1:
                    # for i in np.argwhere(preds["data"]["is_drift"] > 0):
                    #     print(self.count_drift)
                    #     print(preds)
                    #     print(f"Warning detected on {self.features[i]}")
                    return (
                        True,
                        True,
                        np.max(preds["data"]["distance"]),
                        np.min(preds["data"]["p_val"]),
                    )
                else:
                    return (
                        False,
                        True,
                        np.max(preds["data"]["distance"]),
                        np.min(preds["data"]["p_val"]),
                    )
            else:
                self.count_drift = 0
                return (
                    False,
                    False,
                    np.max(preds["data"]["distance"]),
                    np.min(preds["data"]["p_val"]),
                )

    @staticmethod
    def needs_label():
        return False
