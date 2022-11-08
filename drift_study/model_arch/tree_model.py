from mlc.models.sk_models import SkModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestRegressorModel(SkModel):
    def __init__(self, name="rf_regression", **kwargs):
        super().__init__(name=name, objective="regression", **kwargs)
        self.model = RandomForestRegressor(n_jobs=-1)


class RfLcld(SkModel):
    def __init__(self, name="rf_lcld", **kwargs):
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
        self, name="rf_classifier", objective="classification", **kwargs
    ):
        super(Rf, self).__init__(name=name, objective=objective, **kwargs)
        rf_parameters = {
            "class_weight": "balanced",
        }

        self.model = RandomForestClassifier(
            **rf_parameters,
            random_state=kwargs.get("random_state"),
            n_jobs=kwargs.get("n_jobs"),
        )


models = [
    ("rf_regression", RandomForestRegressorModel),
    ("rf_lcld", RfLcld),
    ("rf_classifier", Rf),
]
