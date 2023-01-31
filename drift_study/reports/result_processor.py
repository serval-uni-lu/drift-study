import glob
import os
from typing import Any, Dict, List

from mlc.load_do_save import load_json
from tqdm import tqdm


def load_confs(input_path: str) -> List[Dict[str, Any]]:

    if os.path.isdir(input_path):
        out = []
        list_files = glob.glob(f"{input_path}/*.json")
        for path in tqdm(list_files, total=len(list_files)):
            out.append(load_json(path))
        return out

    if os.path.isfile(input_path):
        return load_json(input_path)

    if os.path.isfile(f"{input_path}.json"):
        return load_json(f"{input_path}.json")

    raise NotImplementedError


def filter_conf(conf_results: Dict[str, Any]) -> Dict[str, Any]:
    detector_name = "_".join(
        [e["name"] for e in conf_results["config"]["runs"][0]["detectors"]]
    )
    metric_name = (
        conf_results["config"]["runs"][0]["detectors"][-1]
        .get("params", {})
        .get("metric_conf", {})
        .get("name")
    )
    if metric_name is not None:
        detector_name = f"{detector_name}_{metric_name}"
    ml_metric = conf_results["ml_metric"][2]
    n_train = conf_results["n_train"][2]

    metric_name = (
        conf_results["config"]
        .get("evaluation_params", {})
        .get("metric", {})
        .get("name", "")
    )
    out = {
        "n_train": n_train,
        "ml_metric": ml_metric,
        "detector_type": conf_results["config"]["runs"][0]["type"],
        "detector_name": detector_name,
        "params": conf_results["config"]["runs"][0]["detectors"][-1].get(
            "params", "No params provided"
        ),
        "metric_name": metric_name,
    }
    return out
