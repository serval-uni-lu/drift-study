import h5py
import numpy as np
from mlc.metrics.metrics import PredClassificationMetric


def score_to_pred(y_score, threshold=None):
    if threshold is None:
        y_pred = np.argmax(y_score, axis=1)
    else:
        y_pred = (y_score[:, 1] >= threshold).astype(int)
    return y_pred


def confusion(y_true, y_pred):

    t = y_true.astype(np.bool)
    f = ~t
    p = y_pred.astype(np.bool)
    n = ~p

    return (
        np.min([t, n], axis=0),
        np.min([f, p], axis=0),
        np.min([f, n], axis=0),
        np.min([t, p], axis=0),
    )


def rolling_sum(a, n):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :]


def rolling_confusion(y_true, y_pred, n):
    tn, fp, fn, tp = confusion(y_true, y_pred)
    tn, fp, fn, tp = (rolling_sum(e, n) for e in (tn, fp, fn, tp))
    return tn, fp, fn, tp


def rolling_f1(y_true, y_pred, n):
    tn, fp, fn, tp = rolling_confusion(y_true, y_pred, n)
    return tp / (tp + 0.5 * (fp + fn))


def load_config_eval(config, dataset, model_name, prediction_metric, y):
    test_i = np.arange(len(y))[config.get("window_size") :]
    batch_size = config.get("evaluation_params").get("batch_size")
    length = len(test_i) - (len(test_i) % batch_size)
    index_batches = np.split(test_i[:length], length / batch_size)
    for run_config in config.get("runs"):
        drift_data_path = (
            f"./data/drift-study/{dataset.name}/{model_name}/"
            f"{run_config.get('name')}.hdf5"
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

    return config
