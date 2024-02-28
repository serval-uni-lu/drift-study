from typing import Any, Dict, Optional, Tuple, Type, Union

import numpy as np
import optuna
import pandas as pd
import torch
from constrained_attacks.attacks.cta.cfab import CFAB
from mlc.constraints.constraints import get_constraints_from_metadata
from mlc.models.model import Model
from mlc.transformers.tab_scaler import TabScaler

from drift_study.drift_detectors.drift_detector import (
    DriftDetector,
    NoModelException,
    NotFittedDetectorException,
)
from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.typing import NDFloat, NDInt, NDNumber


class FabDrift(DriftDetector):
    def __init__(
        self,
        drift_detector: DriftDetector,
        x_metadata,
        **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(
            drift_detector=drift_detector,
            **kwargs,
        )
        self.drift_detector = drift_detector
        self.x_metadata = x_metadata
        self.distance_scaler = None
        self.attack = None
        self.model = None

    def generate_adversarial_examples(self, x: pd.DataFrame, y: NDNumber):
        x_l = torch.from_numpy(x.values).float().to(self.model.device)
        y_l = torch.from_numpy(y).long().to(self.model.device)
        return self.attack.generate(x_l, y_l)

    def adversarial_distance(self, x: pd.DataFrame, y: NDNumber):
        x_adv = self.generate_adversarial_examples(x, y)
        dist = self.distance_scaler.transform(
            x_adv
        ) - self.distance_scaler.transform(x)
        dist = np.linalg.norm(dist, axis=1)
        return dist

    def pred_adversarial_distance(self, x: pd.DataFrame, y_scores: NDFloat):
        y_pred = np.argmax(y_scores, axis=1)
        return self.adversarial_distance(x, y_pred)

    def fit(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
        model: Optional[Model],
    ) -> None:
        x = pd.DataFrame(x, columns=self.x_metadata["feature"])
        if model is None:
            raise NoModelException

        if isinstance(model, LazyPipeline):
            model._pipeline_load()
            model = model.pipeline

        constraints = get_constraints_from_metadata(
            self.x_metadata, relation_constraints=None
        )
        self.model = model
        self.attack = CFAB(
            constraints,
            model.scaler,
            model.wrapper_model,
            model.predict_proba,
            norm="L2",
            eps=10,
            steps=100,
            n_restarts=5,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            verbose=False,
            seed=0,
            multi_targeted=False,
            n_classes=10,
            fix_equality_constraints_end=False,
            fix_equality_constraints_iter=False,
            eps_margin=0.01,
        )
        self.distance_scaler = TabScaler(
            num_scaler="min_max",
            one_hot_encode=True,
        )

        self.distance_scaler.fit_metadata(self.x_metadata)

        distance = self.pred_adversarial_distance(x, y_scores)

        x_new = pd.DataFrame.from_dict({"adv_distance": distance})
        y_scores_new = np.array(distance)
        self.drift_detector.fit(
            x=x_new,
            t=t,
            y=y,
            y_scores=y_scores_new,
            model=model,
        )

    def update(
        self,
        x: pd.DataFrame,
        t: Union[pd.Series, NDInt],
        y: NDNumber,
        y_scores: NDFloat,
    ) -> Tuple[bool, bool, pd.DataFrame]:
        if (self.model is None) or (self.rf_uncertainty is None):
            raise NotFittedDetectorException
        # if not isinstance(self.model, Pipeline):
        #     raise NotImplementedError("Model is expected to be a Pipeline")

        distance = self.pred_adversarial_distance(x, y_scores)

        x_new = pd.DataFrame.from_dict({"adv_distance": distance})
        y_scores_new = np.array(distance)

        is_drift, is_warning, metrics = self.drift_detector.update(
            x=x_new, t=t, y=y, y_scores=y_scores_new
        )
        metrics["adv_distance"] = distance
        return is_drift, is_warning, metrics

    def needs_model(self) -> bool:
        return True

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        return {}


detectors: Dict[str, Type[DriftDetector]] = {"adv_dist_fab": FabDrift}
