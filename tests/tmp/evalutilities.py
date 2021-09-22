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


def ocsvm_anomaly_score(clf, X):  # noqa
    """Return the anomaly score as binary score in {0, 1} (0:inlier, 1:outlier)
    for consistency with the IsolationForest algorithm (see function
    `IsolationForest`)
    """
    # sklearn_svm_OneClassSVM.decision_function returns the signed distance
    # to the separating hyperplane. Signed distance is positive for an
    # inlier and negative for an outlier.
    ret = clf.decision_function(X)
    # OCSVMs do NOT support bounded scores, thus we simpoly need a
    # binary case
    ret[ret >= 0] = 0
    ret[ret < 0] = 1
    return ret


def best_th_roc(y_true, y_pred, *, sample_weight=None):
    """
    Compute the best threshold of a ROC curve computed from an array of class
    labels and relative predictions (or amplitude anomaly scores).
    The best threshold is defined as the threshold maximizing the score:
    `harmonic_mean(1-fpr, tpr)` (fpr=false positive rate, tpr: true positive
    rate. Both quantities are obtained by computing the ROC curve first)

    :param y_true: True binary labels, either in {0, 1} or {True, False}
        (but no mixed types)
    :param y_pred: Target scores (amplitude anomaly scores), as floats in
        [0, 1]

    :return: A the tuple of 4 floats:
        `fpr` is the False positive rate at the best threshold (see below)
        `tpr` is the best True positive rate at the best threshold (see below)
        `threshold` is the best threshold, i.e. the threshold at which the best
            score (see below) is found
        `hmean` is the best score, i.e. the harmonic mean of tpr and tnr
            (tnr=1-fpr)

    """
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred,
                                             sample_weight=sample_weight,
                                             pos_label=1)

    # Convert to TNR (avoid dividing by 2 as useless):
    tnr = 1 - fpr

    # get the best threshold where we have the best mean of TPR and TNR:
    h_mean = harmonic_mean(tnr, tpr)

    # Get the best threshold ignoring 1st score. From the docs
    # (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn.metrics.roc_curve):
    # thresholds[0] represents no instances being predicted and
    # is arbitrarily set to max(y_score) + 1.
    idx = 1 + np.argmax(h_mean[1:])

    # return fpr[idx], tpr[idx], thresholds[idx], h_mean[idx]
    return {
        'roc_curve_best_threshold': thresholds[idx],
        'roc_curve_best_threshold_fpr': fpr[idx],
        'roc_curve_best_threshold_tpr': tpr[idx],
        'roc_curve_best_threshold_hmean_tnr_tpr': h_mean[idx]
    }


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


####################################
# IsolationForest-specific metrics #
####################################


def iforest_ypred(y_true, y_pred):
    """
    Map y_pred to a logistc function in [0, 1] for testing an alternative
    IsolationForest (IF) evaluation approach when performing novelty detection.

    Rationale: by definition, Isolation Forest predictions <=0.5 do not denote any
    distinct anomaly (see original paper).
    Now, consider two instances (`y_true=[0, 1]`, i.e, one normal observation, 0,
    and one anomaly, 1) and two models predicting different  anomaly scores
    (`y_pred`) but producing the same evaluation metric "mean squared error":
    ```
    mean_squared_error(y_true=[0, 1], y_pred=[0.3, 0.6])  # model_1
    mean_squared_error(y_true=[0, 1], y_pred=[0.4, 0.7])  # model_2
    ```
    This is a problem because the second model should be preferable: the gain
    in better prediction of the anomaly (0.7 vs 0.6) should be more remarkable
    than the penalty in the prediction of the normal observation (0.4 vs 0.3),
    which is, according to the paper and in the practice, of little use.

    Now, we can set all scores <=0.5 as 0, thus improving most evaluation
    metrics, but with a step function like this we then would give too much
    penalty to a score of 0.51 wrt 0.5. A continuous function is then preferable.
    We could then map the points (0.5, 0) and (1, 1) linearly, to avoid this
    problem. But then our evaluation would be biased: any normal observation
    with score <=0.5 would "weight" more than any outlier with high score < 1,
    which is always happening: in fact, in previous studies we remarked a score
    distribution between roughly [0.4, 0.8] (scores "away" from endpoints are
    typical of many ensemble methods, the "asymmetry" is most likely more
    inherent to our experiment setup and/or the IF algorithm).

    In account of that, we fixed according to our previous studies a high
    threshold of 0.75, and make the function symmetric: inliers below 0.5 and
    outliers above 0.75 should have the same weight. One good solution
    is the  to create a logistic function that passes through (0.5, 0.1)
    and (0.75, 0.9), and convert the scores given as argument into the relative
    logistic values, still in [0, 1] as the original values
    """
    k = 8 * math.log(9)
    x0 = 5.0 / 8.0
    exponent = k * x0 - k * y_pred
    return np.clip(1. / (1 + math.e ** exponent), 0, 1)  # clip for safety


###########################################################################
# Defining the sklearn.metrics function counterparts for IsolationForest: #
###########################################################################


def median_absolute_error_iforest(y_true, y_pred, *, sample_weight=None):
    y_pred = iforest_ypred(y_true, y_pred)
    return metrics.median_absolute_error(y_true, y_pred, sample_weight=sample_weight)


def mean_absolute_error_iforest(y_true, y_pred, *, sample_weight=None):
    y_pred = iforest_ypred(y_true, y_pred)
    return metrics.mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)


def log_loss_iforest(y_true, y_pred, *, sample_weight=None):
    y_pred = iforest_ypred(y_true, y_pred)
    return metrics.log_loss(y_true, y_pred, sample_weight=sample_weight)


def mean_squared_error_iforest(y_true, y_pred, *, sample_weight=None):
    y_pred = iforest_ypred(y_true, y_pred)
    return metrics.mean_squared_error(y_true, y_pred, sample_weight=sample_weight)


def average_precision_score_iforest(y_true, y_pred, *, sample_weight=None):
    y_pred = iforest_ypred(y_true, y_pred)
    return metrics.average_precision_score(y_true, y_pred, sample_weight=sample_weight)
