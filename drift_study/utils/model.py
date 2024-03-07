from typing import Any, Callable, Dict

import pandas as pd
from mlc.metrics.metric_factory import create_metric
from mlc.models.model import Model
from mlc.models.model_factory import get_model
from mlc.models.pipeline import Pipeline
from mlc.models.sk_models import SkModel

from drift_study.model_arch.lazy_pipeline import LazyPipeline
from drift_study.model_arch.sklearn_opt import TimeOptimizer


def quite_model(model: Model) -> None:
    l_model = model
    if isinstance(l_model, Pipeline):
        l_model = l_model[-1]
    if isinstance(l_model, SkModel):
        l_model.model.set_params(**{"verbose": 0})


def get_f_new_model(
    config: Dict[str, Any],
    metadata_x: pd.DataFrame,
) -> Callable[[], Model]:
    def new_model() -> Model:
        if config.get("optimize", False):
            config["params"] = {
                **config["params"],
                **get_model(config).get_default_params(),
            }
            model = TimeOptimizer(model, create_metric(config.get("metric")))
        else:
            model = get_model_l(config, metadata_x)

        model = LazyPipeline(model)
        return model

    return new_model


def get_model_l(model_config: Dict[str, Any], metadata: pd.DataFrame) -> Model:

    model_class = get_model(model_config)

    model_params = model_config.get("params", {})

    if model_config.get("metadata", False):
        model_params["x_metadata"] = metadata

    model = model_class(**model_params)
    return model
