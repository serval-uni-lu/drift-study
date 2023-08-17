import logging
from copy import deepcopy
from multiprocessing import Manager
from typing import Any, Dict

import configutils
from configutils.utils import ConfigFileParser, merge_parameters
from joblib import Parallel, delayed, parallel_backend

from drift_study import run_simulator
from drift_study.run.no_retrain_baseline import add_best_params_to_model
from drift_study.utils.logging import configure_logger
from drift_study.run.periodic_baseline import run as run_periodic



def create_config_params(config: Dict[str, Any], train_window_size: int):
    config["train_window_size"] = train_window_size
    config["schedule_data_path"] = config["schedule_data_path"] + f"/{train_window_size}"
    return config
    
    
def run(config: Dict[str, Any]):
    
    config_to_run = [
        create_config_params(deepcopy(config), train_window_size) 
        for train_window_size in config.get("train_window_sizes")
    ]
    
    for c in config_to_run:
        run_periodic(c)
    
if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
