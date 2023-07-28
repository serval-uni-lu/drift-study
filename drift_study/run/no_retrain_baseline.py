from typing import Any, Dict

import configutils
from configutils.utils import ConfigFileParser, merge_parameters

from drift_study.run_simulator import run as simulator_run


def add_best_params_to_models(config: Dict[str, Any]) -> Dict[str, Any]:
    for model in config["models"]:
        path = (
            f"./data/drift/{config.get('dataset', {}).get('name')}/"
            f"{model.get('name')}/model_opt/best_params.json"
        )
        model_params = ConfigFileParser().do(path)
        model["params"] = merge_parameters(
            model_params, model.get("params", {})
        )

    return config


def run(auto_config: Dict[str, Any]) -> None:

    # Merge the auto config
    config_to_run = merge_parameters(
        auto_config, ConfigFileParser().do("./config/auto/no_detection.yaml")
    )
    # If we want to use the best parameters found by the auto tuner

    if auto_config.get("use_auto_model_tuning"):
        config_to_run = add_best_params_to_models(config_to_run)

    configs_to_run = []
    for model in config_to_run["models"]:
        current_conf = config_to_run.copy()
        current_conf.pop("models")
        current_conf["model"] = model
        configs_to_run.append(current_conf)

    for c in configs_to_run:
        simulator_run(c)


if __name__ == "__main__":
    auto_config = configutils.get_config()
    run(auto_config)