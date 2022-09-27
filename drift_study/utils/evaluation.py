import numpy as np


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
