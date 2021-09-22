"""
Module implementing custom functions for the evaluation routine
"""
import math

import numpy as np
from sklearn import metrics
from scipy.stats import hmean


def isf_anomaly_score(clf, X):  # noqa
    """Return the anomaly score from an IsolationForest classifier `clf`
    as in the original paper, bypassing sklearn conversions to binary [-1,1]
    """
    # sklearn_ensemble_IsolationForest has score_samples that returns the
    # inverse anomaly score of the original paper. So:
    return -clf.score_samples(X)


def best_th_prc(y_true, y_pred, *, sample_weight=None):
    """
    Compute the best threshold of a Precision-Recall curve computed from an
    array of class  labels and relative predictions (or amplitude anomaly
    scores).
    The best threshold is defined as the threshold maximizing the F1-score:
    `harmonic_mean(pre, rec)` (pre=Precision, rec=Recall. Both quantities are
    obtained by computing the Precision-Recall curve first)

    :param y_true: True binary labels, either in {0, 1} or {True, False}
        (but no mixed types)
    :param y_pred: Target scores (amplitude anomaly scores), as floats in
        [0, 1]

    :return: A the tuple of 4 floats:
        `pre` is the Precision at the best threshold (see below)
        `rec` is the Recall at the best threshold (see below)
        `threshold` is the best threshold, i.e. the threshold at which the best
            score (see below) is found
        `best_score` is the best score, i.e. the harmonic mean of p and r, also
            known as F1score
    """
    pre, rec, thresholds = metrics.precision_recall_curve(y_true, y_pred,
                                                          sample_weight=sample_weight,
                                                          pos_label=1)

    # get the best threshold where we have the best F1 score
    # (avoid multiplying by 2 as useless):
    # also , consider avoiding numpy warning for NaNs:
    f1scores = harmonic_mean(pre, rec)

    # Get best score ignoring lat score. From the docs
    # (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html#sklearn.metrics.precision_recall_curve):
    # the last precision and recall values are 1. and 0. respectively and
    # do not have a corresponding threshold. This ensures that the graph
    # starts on the y axis.
    idx = np.argmax(f1scores[:-1])

    # return pre[idx], rec[idx], thresholds[idx], f1scores[idx]
    return {
        'pr_curve_best_threshold': thresholds[idx],
        'pr_curve_best_threshold_precision': pre[idx],
        'pr_curve_best_threshold_recall': rec[idx],
        'pr_curve_best_threshold_f1score': f1scores[idx]
    }


def harmonic_mean(x, y):
    """Wrapper around `scipy.hmean`: compute the harmonic mean of x and y
    element-wise, returning an array of floats in [0,1] the same length of x
    and y. NaNs in either x or y will result in NaNs with no exception raised
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    isfinite = np.isfinite(x) & np.isfinite(y)
    if isfinite.all():
        return hmean([x, y], axis=0)
    else:
        scores = np.empty(len(x))
        scores.fill(np.nan)
        if isfinite.any():
            scores[isfinite] = hmean(np.array([x[isfinite], y[isfinite]]), axis=0)
        return scores
