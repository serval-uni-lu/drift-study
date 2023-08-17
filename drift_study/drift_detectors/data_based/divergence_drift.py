import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
from mlc.models.model import Model
from scipy.spatial.distance import jensenshannon

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NotFittedDetectorException,
)
from drift_study.typing import NDFloat, NDInt, NDNumber

warnings.simplefilter(action="ignore", category=FutureWarning)


class DivergenceDrift(DriftDetector):
    def __init__(
        self,
        x_metadata: pd.DataFrame,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        num_threshold: float = 0.05,
        cat_threshold: float = 0.05,
        n_bin: int = 20,
        drift_share: float = 0.5,
        **kwargs: Dict[str, Any],
    ) -> None:
        self.x_metadata = x_metadata
        super().__init__(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            num_threshold=num_threshold,
            cat_threshold=cat_threshold,
            drift_share=drift_share,
            **kwargs,
        )
        self.window_size = 0
        self.numerical_features = (
            numerical_features if numerical_features is not None else []
        )
        self.categorical_features = (
            categorical_features if categorical_features is not None else []
        )
        self.drift_share = drift_share
        self.num_threshold = num_threshold
        self.cat_threshold = cat_threshold
        self.n_bin = n_bin

        self.fitted = False

        self.x_ref = pd.DataFrame()
        self.x_last = pd.DataFrame()
        self.num_hists: Dict[str, NDNumber] = {}
        self.num_bins: Dict[str, NDNumber] = {}
        self.cat_hists: Dict[str, NDNumber] = {}

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: NDNumber,
        y_scores: NDFloat,
        model: Optional[Model],
    ) -> None:

        if isinstance(x, np.ndarray):
            x = pd.DataFrame(x, columns=self.x_metadata["feature"])

        self.x_ref = x.copy()
        self.x_last = x.copy()
        self.window_size = len(x)

        for name in self.numerical_features:
            self.num_hists[name], self.num_bins[name] = np.histogram(
                x[name].to_numpy(), bins=self.n_bin
            )

        # Categorical features
        for name in self.categorical_features:
            self.cat_hists[name] = np.bincount(x[name].to_numpy().astype(int))

        self.fitted = True

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
    ) -> Tuple[bool, bool, pd.DataFrame]:
        x = pd.DataFrame(x, columns=self.x_metadata["feature"])
        self.x_last = pd.concat([self.x_last, x], ignore_index=True)
        self.x_last = self.x_last.iloc[-self.window_size :]

        return self.evaluate(self.x_last, t, y, y_scores)

    def evaluate(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
    ) -> Tuple[bool, bool, pd.DataFrame]:
        # Numerical features

        drifted = []

        for name in self.numerical_features:
            ref = self.num_hists[name]
            cur, _ = np.histogram(x[name].to_numpy(), bins=self.num_bins[name])
            metric = jensenshannon(ref, cur)
            self.drift_logger.log_metric(
                f"jsd_num_{name}", metric, 1, 1, time.time()
            )
            drifted.append(metric >= self.num_threshold)

        # Categorical features
        for name in self.categorical_features:
            ref = self.cat_hists[name]
            cur = np.bincount(
                x[name].to_numpy().astype(int), minlength=len(ref)
            )[: len(ref)]
            metric = jensenshannon(ref, cur)
            self.drift_logger.log_metric(
                f"jsd_cat_{name}", metric, 1, 1, time.time()
            )
            drifted.append(metric >= self.cat_threshold)

        drifted_mean = np.mean(drifted)
        print(drifted_mean)
        is_drift = drifted_mean >= self.drift_share
        self.drift_logger.log_metric(
            "jsd_drift_share", drifted_mean, 1, 1, time.time()
        )
        self.drift_logger.log_metric(
            "jsd_is_drift", is_drift, 1, 1, time.time()
        )

        return is_drift, is_drift, None

    def needs_model(self) -> bool:
        return False

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "cat_threshold": trial.suggest_float(
                "categorical_threshold", 1e-6, 0.5, log=True
            ),
            "num_threshold": trial.suggest_float(
                "numerical_threshold", 1e-6, 0.5, log=True
            ),
            "drift_share": trial.suggest_float(
                "drift_share",
                1 / trial_params["n_features"],
                1,
                step=1 / trial_params["n_features"],
            ),
        }

    @staticmethod
    def get_default_params(
        trial_params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        return {
            "numerical_threshold": 0.05,
            "categorical_threshold": 0.05,
            "drift_share": 0.5,
        }

    def fit_from_cache(self) -> None:
        if self.fit_cache is None:
            raise NotFittedDetectorException

        cache = self.fit_cache.load()

        def convert(x: pd.DataFrame) -> Dict[str, NDNumber]:
            return dict(zip(x.columns, x.values.T))

        self.cat_hists = convert(cache["cat_hists"])
        self.num_hists = convert(cache["num_hists"])
        self.num_bins = convert(cache["num_bins"])

        # remove nan from self.cat_hists
        self.cat_hists = {
            k: v[~np.isnan(v)] for k, v in self.cat_hists.items()
        }
        self.fitted = True

    def save_cache(self) -> None:
        # Numerical
        cache = {}
        for name, v in [
            ("num_hists", self.num_hists),
            ("num_bins", self.num_bins),
        ]:
            cache[name] = pd.DataFrame(v)

        # Categorical
        categorical = np.empty(
            (
                np.max([len(v) for v in self.cat_hists.values()]),
                len(self.cat_hists),
            )
        )
        categorical[:] = np.nan
        for i, w in enumerate(self.cat_hists.values()):
            categorical[: len(w), i] = w

        cache["cat_hists"] = pd.DataFrame(
            categorical, columns=self.cat_hists.keys()
        )

        self.fit_cache.save(cache)


detectors: Dict[str, Type[DriftDetector]] = {"divergence": DivergenceDrift}
