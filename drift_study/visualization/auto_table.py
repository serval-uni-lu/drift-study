from typing import Any, Dict, Tuple
from drift_study.visualization.plot_auto import get_config_metrics
import pandas as pd
import configutils

def to_df(config_metrics: Tuple[Dict[str, Any], Dict[str, Any]]) -> pd.DataFrame:
    
    dict_df = {
        "period": [],
        "train_window_size": [],
        "ml_metric": []
    }
    for e in config_metrics:
        conf = e[0]
        metric = e[1]
        
        dict_df["period"].append(conf["schedule"]["detectors"][0]["params"]["period"])
        # dict_df["period"] = conf["schedule"]["detectors"][0]["params"]["period"]
        dict_df["train_window_size"].append(conf["train_window_size"])
        dict_df["ml_metric"].append(metric["ml_metric"])
        
    return pd.DataFrame.from_dict(dict_df)

    

def run(config: Dict[str, Any]):
    
    schedule_data_path = config["schedule_data_path"]
    
    paths = [f"{schedule_data_path}/{w}/periodic_{p}" for w in config.get("train_window_sizes") for p in config.get("periods")]
    print(paths)
    config_metrics = get_config_metrics(paths)
    
    df = to_df(config_metrics)
    table = pd.pivot_table(df, values="ml_metric", index=["period"], columns=['train_window_size'])
    table.to_csv(config["out_path"])
    

if __name__ == "__main__":
    config = configutils.get_config()
    run(config)
