import time
import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import numpy.typing as npt
import optuna
import pandas
import pandas as pd
from mlc.models.model import Model

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NoModelException,
    NotFittedDetectorException,
)
from drift_study.typing import NDFloat, NDInt, NDNumber

from .rf_uncertainty import RandomForestClassifierWithUncertainty

# from drift_prod.model_arch.lazy_pipeline import LazyPipeline


warnings.simplefilter(action="ignore", category=FutureWarning)

UNCERTAINTY_TYPE = {"total": 0, "epistemic": 1, "aleatoric": 2}


class RfUncertaintyDrift(DriftDetector):
    def __init__(
        self,
        drift_detector: DriftDetector,
        x_metadata,
        uncertainty_type: str,
        rf: Optional[Model] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            drift_detector=drift_detector,
            uncertainty_type=uncertainty_type,
            **kwargs,
        )
        self.drift_detector = drift_detector
        self.uncertainty_type = uncertainty_type
        self.rf_uncertainty: Optional[
            RandomForestClassifierWithUncertainty
        ] = None
        self.rf = rf
        self.model: Optional[Model] = None
        self.x_metadata = x_metadata

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
        model: Optional[Model],
    ) -> None:
        x = pd.DataFrame(x, columns=self.x_metadata["feature"])
        if model is None:
            raise NoModelException

        # if isinstance(model, LazyPipeline):
        #     model._pipeline_load()
        #     model = model.pipeline

        # if not isinstance(model, Pipeline):
        #     raise NotImplementedError("Model is expected to be a Pipeline")

        internal_model = model
        while hasattr(internal_model, "model"):
            internal_model = internal_model.model
        self.rf = internal_model
        self.model = model

        self.rf_uncertainty = RandomForestClassifierWithUncertainty(
            random_forest=self.rf
        )
        self.rf_uncertainty.fit(x, y)
        (
            _,
            uncertainties,
        ) = self.rf_uncertainty.predict_proba_with_uncertainty(x)
        uncertainties = uncertainties[UNCERTAINTY_TYPE[self.uncertainty_type]]
        # x_new = pd.DataFrame.from_dict({"uncertainty": uncertainties})
        # y_scores_new = np.array(uncertainties)
        # self.drift_detector.fit(
        #     x=x_new,
        #     t=t,
        #     y=y,
        #     y_scores=y_scores_new,
        #     model=model,
        # )

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, npt.NDArray[np.int_]],
        y: Union[npt.NDArray[np.int_], npt.NDArray[np.float_]],
        y_scores: Union[npt.NDArray[np.float_]],
    ) -> Tuple[bool, bool, pd.DataFrame]:
        if (self.model is None) or (self.rf_uncertainty is None):
            raise NotFittedDetectorException
        # if not isinstance(self.model, Pipeline):
        #     raise NotImplementedError("Model is expected to be a Pipeline")
        x = pandas.DataFrame(x)
        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            self.model.model.scaler.transform(x)
        )

        uncertainties = uncertainties[UNCERTAINTY_TYPE[self.uncertainty_type]]
        x_new = pd.DataFrame.from_dict({"uncertainty": uncertainties})
        y_scores_new = np.array(uncertainties)

        is_drift, is_warning, metrics = self.drift_detector.update(
            x=x_new, t=t, y=y, y_scores=y_scores_new
        )
        metrics[f"{self.uncertainty_type}_uncertainty"] = uncertainties
        return is_drift, is_warning, metrics

    def evaluate(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
    ) -> Tuple[bool, bool, pd.DataFrame]:
        # Numerical features

        _, uncertainties = self.rf_uncertainty.predict_proba_with_uncertainty(
            x
        )

        for name, metric in zip(UNCERTAINTY_TYPE.keys(), uncertainties):
            self.drift_logger.log_metric(
                f"uncertainty_{name}", np.mean(metric), 1, 1, time.time()
            )

        return False, False, None

    def needs_model(self) -> bool:
        return True

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "uncertainty_type": trial.suggest_categorical(
                "uncertainty_type", ["total", "epistemic", "aleatoric"]
            )
        }

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        return {"uncertainty_type": "total"}

    def fit_from_cache(self) -> None:
        if self.fit_cache is None:
            raise NotFittedDetectorException
        leafs_content_df = self.fit_cache.load()["leafs_content"]
        labels = self.fit_cache.load()["labels"]

        self.rf_uncertainty = RandomForestClassifierWithUncertainty(
            random_forest=self.rf
        )
        self.rf_uncertainty._labels = labels["labels"].to_numpy()

        leafs_content = []
        last_tree = -1
        for index, row in leafs_content_df.iterrows():
            if row["tree"] != last_tree:
                last_tree = row["tree"]
                leafs_content.append({})

            leafs_content[-1][row["node"]] = [
                row[f"label_{i}"]
                for i in range(len(self.rf_uncertainty._labels))
            ]

        self.rf_uncertainty.leafs_content = leafs_content

    def save_cache(self) -> None:

        # Leaf

        leaf_content = {"tree": [], "node": []}
        n_labels = len(self.rf_uncertainty._labels)
        for i in range(n_labels):
            leaf_content[f"label_{i}"] = []

        for tree_i, tree in enumerate(self.rf_uncertainty.leafs_content):
            for node_i, node_value in tree.items():
                leaf_content["tree"].append(tree_i)
                leaf_content["node"].append(node_i)
                for i in range(n_labels):
                    leaf_content[f"label_{i}"].append(node_value[i])

        cache = {
            "leafs_content": pd.DataFrame.from_dict(leaf_content),
            "labels": pd.DataFrame(
                self.rf_uncertainty._labels, columns=["labels"]
            ),
        }

        self.fit_cache.save(cache)

        # a = [
        #     {int(node_i), node_v.list()}
        #     for node_i, node_v in tree.items()
        #     for tree in self.rf_uncertainty.leafs_content
        # ]


detectors: Dict[str, Type[DriftDetector]] = {
    "rf_uncertainty": RfUncertaintyDrift
}
