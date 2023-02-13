from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from mlc.datasets.dataset import Dataset
from mlc.metrics.metric import Metric
from mlc.metrics.metrics import PredClassificationMetric


def score_to_pred(
    y_score: npt.NDArray[np.float_], threshold: Optional[float] = None
) -> npt.NDArray[np.int_]:
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(int)
    return y_pred


def confusion(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_]
) -> Tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
]:

    t = y_true.astype(np.bool_)
    f = ~t
    p = y_pred.astype(np.bool_)
    n = ~p

    return (
        np.min([t, n], axis=0),
        np.min([f, p], axis=0),
        np.min([f, n], axis=0),
        np.min([t, p], axis=0),
    )


def rolling_sum(a: npt.NDArray[Any], n: int) -> npt.NDArray[Any]:
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :]


def rolling_confusion(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_], n: int
) -> Tuple[
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
    npt.NDArray[np.int_],
]:
    tn, fp, fn, tp = confusion(y_true, y_pred)
    tn, fp, fn, tp = (rolling_sum(e, n) for e in (tn, fp, fn, tp))
    return tn, fp, fn, tp


def rolling_f1(
    y_true: npt.NDArray[np.int_], y_pred: npt.NDArray[np.int_], n: int
) -> npt.NDArray[np.float_]:
    tn, fp, fn, tp = rolling_confusion(y_true, y_pred, n)
    return tp / (tp + 0.5 * (fp + fn))


def load_config_eval(
    config: Dict[str, Any],
    dataset: Dataset,
    prediction_metric: Metric,
    y: npt.NDArray[Any],
) -> Dict[str, Any]:
    test_i = np.arange(len(y))[config["evaluation_params"]["end_train_idx"] :]
    batch_size = config["evaluation_params"]["batch_size"]
    batch_size_min = config["evaluation_params"].get("batch_size", 0)
    last_idx = config["evaluation_params"].get("last_idx", -1)
    if last_idx > -1:
        test_i = test_i[: last_idx - config.get("window_size")]
    if isinstance(batch_size, int):
        length = len(test_i) - (len(test_i) % batch_size)
        index_batches = np.split(test_i[:length], int(length / batch_size))
    if isinstance(batch_size, str):

        d = dataset.get_x_y_t()[2].iloc[test_i]
        d = pd.DataFrame(d, columns=["DATE"])
        d = d.groupby(pd.Grouper(key="DATE", axis=0, freq=batch_size)).size()

        index_batches = []
        for e in d:
            if e >= batch_size_min:
                np.concatenate(np.arange(last_idx, last_idx + e))
            last_idx += e

    sub_dir_path = config["sub_dir_path"]
    for run_config in config.get("runs", []):
        model_name = run_config.get("model").get("name")
        drift_data_path = (
            f"./data/simulator/{dataset.name}/{model_name}/"
            f"{sub_dir_path}/{run_config.get('name')}.hdf5"
        )
        with h5py.File(drift_data_path, "r") as f:
            y_scores = f["y_scores"][()]
            model_used = f["model_used"][()]

        # Check if retrained
        run_config["model_used"] = model_used
        run_config["is_retrained"] = []
        for i, index_batch in enumerate(index_batches):
            if i == 0:
                run_config["is_retrained"].append(True)
            else:
                if (
                    model_used[index_batch].max()
                    != model_used[index_batches[i - 1]].max()
                ):
                    run_config["is_retrained"].append(True)
                else:
                    run_config["is_retrained"].append(False)

        if isinstance(prediction_metric, PredClassificationMetric):
            y_scores = np.argmax(y_scores, axis=1)

        run_config["prediction_metric"] = prediction_metric.compute(
            y[test_i], y_scores[test_i]
        )
        run_config["prediction_metric_batch"] = np.array(
            [
                prediction_metric.compute(
                    y[index_batch], y_scores[index_batch]
                )
                for index_batch in index_batches
            ]
        )
        run_config["batch_start_idx"] = np.array([e[0] for e in index_batches])

    return config
