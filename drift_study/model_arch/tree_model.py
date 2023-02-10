from math import sqrt
from typing import Any, Dict

import optuna
from mlc.metrics.metric_factory import create_metric
from mlc.models.sk_models import SkModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from drift_study.model_arch.sklearn_opt import TimeOptimizer


class RandomForestRegressorModel(SkModel):
    def __init__(self, name: str = "rf_regression", **kwargs: Any):
        super().__init__(name=name, objective="regression", **kwargs)
        self.model = RandomForestRegressor(n_jobs=-1)


class RfLcld(SkModel):
    def __init__(self, name: str = "rf_lcld", **kwargs: Any):
        super(RfLcld, self).__init__(name=name, objective="binary", **kwargs)
        rf_parameters = {
            "n_estimators": 125,
            "min_samples_split": 6,
            "min_samples_leaf": 2,
            "max_depth": 10,
            "bootstrap": True,
            "class_weight": "balanced",
        }

        self.model = RandomForestClassifier(
            **rf_parameters,
            random_state=kwargs.get("random_state"),
            n_jobs=kwargs.get("n_jobs"),
        )


class Rf(SkModel):
    def __init__(
        self,
        name: str = "rf_classifier",
        objective: str = "classification",
        **kwargs: Any,
    ):
        super().__init__(name=name, objective=objective, **kwargs)
        rf_parameters = {"class_weight": "balanced", **kwargs}

        self.model = RandomForestClassifier(
            **rf_parameters,
        )

    @staticmethod
    def define_trial_parameters(
        trial: optuna.Trial, trial_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "max_depth": trial.suggest_int("max_depth", 4, 50),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 150
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 60),
            "max_features": trial.suggest_float("max_features", 0, 1),
        }

        return params

    @staticmethod
    def get_default_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "n_estimators": 100,
            "max_depth": 50,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": (
                sqrt(trial_params["n_features"]) / trial_params["n_features"]
            ),
        }

        return params


models = [
    ("rf_regression", RandomForestRegressorModel),
    ("rf_lcld", RfLcld),
    ("rf_classifier", Rf),
    (
        "rf_opt_mcc",
        lambda **kwargs: TimeOptimizer(
            model=Rf(),
            metric=create_metric({"name": "mcc"}),
            n_trials=25,
            n_splits=5,
            **kwargs,
        ),
    ),
]
