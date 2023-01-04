import glob
from typing import Any, Dict

import configutils
import pandas as pd
from joblib import Parallel, delayed
from mlc.load_do_save import load_json
from tqdm import tqdm


def filter_conf(conf_results: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "n_train": conf_results["n_train"],
        "ml_metric": conf_results["ml_metric"],
        "run_name": conf_results["config"]["runs"][0]["name"],
        "params": conf_results["config"]["runs"][0]["detectors"][-1].get(
            "params", "No params provided"
        ),
    }
    return out


def run() -> None:
    config = configutils.get_config()
    print(config)

    input_dir = config["input_dir"]
    n_jobs = config.get("n_jobs", 1)

    list_files = glob.glob(f"{input_dir}/*.json")
    conf_results = Parallel(n_jobs=n_jobs)(
        delayed(lambda x: filter_conf(load_json(x)))(path)
        for path in tqdm(list_files, total=len(list_files))
    )
    df = pd.DataFrame(conf_results)
    print(df)


if __name__ == "__main__":
    run()
