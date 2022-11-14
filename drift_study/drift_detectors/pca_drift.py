import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from drift_study.drift_detectors import EvidentlyDrift
from drift_study.drift_detectors.pca_cd import PCACD


class PcaDrift:
    def __init__(
        self,
        variance,
        drift_detector,
        **kwargs,
    ) -> None:
        self.drift_detector = drift_detector
        self.variance = variance
        self.pca_transformer = None

    def fit(self, x, **kwargs):
        self.pca_transformer = PCA(
            n_components=self.variance, svd_solver="full"
        )
        x = self.pca_transformer.fit_transform(x)
        x = pd.DataFrame(x, columns=np.arange(x.shape[1]))
        if isinstance(self.drift_detector, EvidentlyDrift):
            self.drift_detector.categorical_features = None
            self.drift_detector.numerical_features = x.columns
        self.drift_detector.fit(x=x, **kwargs)

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)
        return self.drift_detector.update(x=self.pca_transformer.transform(x))

    @staticmethod
    def needs_label():
        return False


class PcaCdDrift:
    def __init__(
        self,
        window_size,
        **kwargs,
    ) -> None:
        self.window_size = window_size
        self.detector = None

    def fit(self, x, **kwargs):
        self.detector = PCACD(
            window_size=self.window_size,
            online_scaling=True,
            ev_threshold=0.999999,
            divergence_metric="intersection",
        )
        self.detector.update(x)
        self.detector.update(x)

    def update(self, x, **kwargs):
        x = pd.DataFrame(x)
        # self.detector.update(x.to_numpy().reshape(1, -1))
        out = self.detector.update(x)
        if out is None:
            return False, False, pd.DataFrame()
        return out

    @staticmethod
    def needs_label():
        return False
