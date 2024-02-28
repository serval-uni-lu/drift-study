import glob

import configutils
import h5py
import numpy as np
from mlc.load_do_save import load_json
from tqdm import tqdm

from drift_study.reports.result_processor import load_confs


def run(input_dir, input_dir_opt, input_dir_sim) -> None:
    config = configutils.get_config()
    print(config)

    # input_dir = config["input_dir"]
    # input_dir_opt = config["input_dir_opt"]
    # input_dir_sim = config["input_dir_sim"]

    conf_results = load_confs(input_dir)
    conf_names = [e["config"]["runs"][0]["name"] for e in conf_results]
    val_idx = conf_results[0]["config"]["evaluation_params"]["val_test_idx"]

    conf_results_opt = []
    sims = []
    for e in conf_names:
        conf_results_opt.append((load_json(f"{input_dir_opt}/{e}.json")))
        with h5py.File(f"{input_dir_sim}/{e}.hdf5", "r+") as f:
            a = f["model_used"][()]
            sims.append(np.unique(a[val_idx:]))

    not_ok = []
    for i in range(len(conf_results)):

        a = conf_results[i]["n_train"] - conf_results_opt[i]["n_train"] + 1
        b = len(sims[i])
        if a != b:
            print(
                a,
                b,
                conf_names[i],
                sims[i],
                conf_results[i]["n_train"],
                conf_results_opt[i]["n_train"],
            )
            not_ok.append(conf_names[i])

    print(len(not_ok))


def run_many():

    USE_CASES = [
        (
            "lcld_201317_ds_time/rf_lcld",
            "all_delays",
            "small_all_delays6",
            "all_delays",
        ),
        (
            "lcld_201317_ds_time/rf_lcld",
            "all_delays_half",
            "small_all_delays6",
            "all_delays_half",
        ),
    ]
    for e in USE_CASES:
        run(
            f"./data/complete_run/{e[0]}/{e[1]}/",
            f"./data/optimizer_results/{e[0]}/{e[2]}/",
            f"./data/simulator/{e[0]}/{e[3]}/",
        )


if __name__ == "__main__":
    run_many()
