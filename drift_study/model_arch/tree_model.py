from mlc.models.sk_models import SkModel
from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressorModel(SkModel):
    def __init__(self, name="rf_regression", **kwargs):
        super().__init__(name=name, objective="regression", **kwargs)
        self.model = RandomForestRegressor(n_jobs=-1)


models = [
    ("rf_regression", RandomForestRegressorModel),
]
