from copy import deepcopy
from typing import Any, Dict

import configutils
from configutils.utils import ConfigFileParser, merge_parameters

from drift_study.run_simulator import run as simulator_run


def add_best_params_to_model(config: Dict[str, Any]) -> Dict[str, Any]:
    model = config["model"]
    path = (
        f"./data/drift/{config.get('dataset', {}).get('name')}/"
        f"{model.get('name')}/model_opt/best_params.json"
    )
    model_params = ConfigFileParser().do(path)
    model["params"] = merge_parameters(model_params, model.get("params", {}))

    return config


def run(auto_config: Dict[str, Any]) -> None:

    # Merge the auto config
    config_to_run = merge_parameters(
        auto_config, ConfigFileParser().do("./config/auto/manual.yaml")
    )
    config_to_run = merge_parameters(
        config_to_run, ConfigFileParser().do("./config/auto/delays_none.yaml")
    )
    detect_idxs = [e - 1 for e in config_to_run["manual_retrain"]]
    print(detect_idxs)
    config_to_run["schedule"]["detectors"][0]["params"] = {
        "detect_idxs": detect_idxs
    }

    # If we want to use the best parameters found by the auto tuner

    if auto_config.get("use_auto_model_tuning"):
        config_to_run = add_best_params_to_model(config_to_run)

    # config_opt = deepcopy(config_to_run)
    # config_opt["model"]["optimize"] = True
    # config_opt["schedule_data_path"] = (
    #     config_opt["schedule_data_path"].rstrip("/") + "_opt/"
    # )
    simulator_run(config_to_run)
    # simulator_run(config_opt)


if __name__ == "__main__":
    auto_config = configutils.get_config()
    run(auto_config)
