"""Scikit and pandas core utilities that can be used in custom code"""

import numpy as np
import pandas as pd


def dropna(dataframe, features=None, inf_is_na=False):
    """
    Same as `dataframe.dropna(subset=features)`, optionally removing
    also rows with +-inf values
    """
    # Use `isna` as single-point of definition and let's skip `dataframe.dropna`:
    return dataframe[~isna(dataframe, features, inf_is_na)]


def isna(dataframe, features=None, inf_is_na=False):
    """
    Return an array of booleans the same length of the given dataframe,
    indicating rows with at least one NA (not available, or missing) value, i.e.
    None or NaN (and +-inf if `inf_is_na` is True)

    :param features: list, string or None. The features (dataframe columns)
        to consider. None (the default) means: all columns. If string, it
        denotes the (only) column to consider
    :param inf_is_na: boolean (default False), tells if +-inf are considered
        NA
    """
    dataframe = dataframe if features is None else dataframe[features]
    with pd.option_context('mode.use_inf_as_na', inf_is_na):
        return dataframe.isna().any(axis=1)


def classifier(clf_class, clf_parameters, trainingset, features=None, drop_na=True,
               inf_is_na=True, **fit_kwargs):
    """
    Return a scikit learn model fitted with the data of `trainingset`.
    See :func:`save_clf` and :func:`load_clf` for serializing the created
    classifier

    :param clf: a class denoting a sklearn classifier (e.g.
        :class:`sklearn.ensemble.IsolationForest`)
    :param clf_parameters: the parameters of the classifier `__init__` method
    :param trainingset: pandas DataFrame denoting the training set
        (rows:instances, columns: features)
    :param features: list[str] or str. The features to train/fit the classifier
        with. It must be a list of columns of the training and test sets.
        If None (the default), all columns will be used. If string, it denotes
        the only column to be used
    :param drop_na: boolean. Drop instances from the training set that have
        any NA/missing value, before fitting the classifier. NA values are
        either NaN or None (and optionally +-Inf, see `inf_is_na`).
        Set this argument to False only if the classifier handles NaNs
    :param inf_is_na: boolean (default: True), tells if +-Inf have to be
        considered NA. This argument is ignored if `drop_na` is False
    :param fit_kwargs: optional keyword arguments for the classifier `fit` method

    :return: a fitted classifier object
    """
    clf = clf_class(**clf_parameters)
    tr_set = trainingset
    if features is not None:
        if isinstance(features, str):
            features = [features]
        tr_set = trainingset[features]
    if drop_na:
        tr_set = dropna(tr_set, inf_is_na=inf_is_na).copy()
    clf.fit(tr_set.values.copy(), **fit_kwargs)
    return clf


def evaluate(clf, testset, y_true, prediction_function,
             evaluation_metrics, features=None,
             sample_weight=None, drop_na=True, inf_is_na=True):
    """
    Evaluate the given pre-trained classifier by predicting
    the scores of the given test set(s) and returning the given evaluation
    metrics

    :param clf: a pre-trained classifier object (e.g. an instance of
        :class:`sklearn.ensemble.IsolationForest`)
    :param testset: pandas dataframe denoting the test set(s)
        (rows:instances, columns: features)
    :param features: features to train/fit the classifier with, and to
        be used to predict test set(s) data. It must be a list of columns of
        the training and test sets. If None (the default), all columns will be
        used (except `y_true` and `sample_weight`, if strings. See below).
        For a single feature, you can input a simple string instead of a
        1-element list
    :param y_true: string or numeric/boolean array. The true
    `   class/score/probability, one element per test set row. If string, it
        must be a column of the test set
    :param prediction_function: a function of signature `(clf, X)` returning
        the prediction or classification of the instances of the given test
        set X (matrix of N instances X M features). It can be a method of
        `clf` such as `decision_function` or `predict` (see sklearn docs) or
        a user defined function
    :param evaluation_metrics: Callable, string, or iterable of Callable/strings.
        Any string must denote a Python path to an evaluation function that will
        be dynamically loaded. Eventually, all functions must have signature
        `(y_true, y_pred, sample_weight=None, ...)` and must return either a
        numeric scalar, or a dict of metric names (string) mapped to their
        numeric values
    :param sample_weight: string or numeric array, default None. The test set
        column to be used as sample weight in the evaluation metrics
        (`sample_weight` argument). If string, it must be a numeric column
        of the test set(s)
    :param drop_na: boolean. Drop or ignore instances that have any NA/missing
        value. NA values are either NaN or None (and optionally +-Inf, see
        `inf_is_na`).
        NA instances will be dropped from the training set and ignored in the
        test set (their prediction will be set as `numpy.nan`) and
        consequently, in all evaluation metrics.
        Set this argument to False only if the classifier handle NaNs, i.e. it
        always it always produces scores/predictions not NA
    :param inf_is_na: boolean (default: True), tells if +-Inf have to be
        considered NA. This argument is ignored if `drop_na` is False

    :return: the tuple `(y_pred, evaluation)` denoting:
        `y_pred` (numpy numeric array): the predicted scores/classes/probabilities
            of the test set instances (element/row-wise) and
        `evaluation` (dict): each computed evaluation metric name, mapped to
            its value
    """
    non_feature_columns = set()
    if isinstance(y_true, str):
        non_feature_columns.add(y_true)
        y_true = testset[y_true]
    if isinstance(sample_weight, str):
        non_feature_columns.add(sample_weight)
        sample_weight = testset[sample_weight]

    if features is None:
        features = [f for f in testset.columns if f not in non_feature_columns]
    y_pred = predict(clf, prediction_function, testset, features, drop_na,
                     inf_is_na=inf_is_na, inplace=False)

    evaluation = compute_metrics(evaluation_metrics, y_true, y_pred,
                                 sample_weight=sample_weight)

    return y_pred, evaluation


def predict(clf, prediction_function, testset, features=None, drop_na=True,
            inf_is_na=True, inplace=False):
    """
    Predicts the scores/classes/probabilities of test-set data frame.

    :param clf: the classifier object
    :param prediction_function: a function of signature `(clf, X)` returning
        the prediction or classification of the instances  of the test set X
        (matrix of N instances X M features). It can be a method of `clf` such
        as `decision_function` or `predict` (see sklearn docs) or a user
        defined function
    :param testset: pandas DataFrame, one instance per row, one
        feature per column
    :param features: list[str] or str. The column(s) of the dataframe denoting
        the  features to be used, i.e. the features that `clf` has been trained
        with. If None (the default) all columns are used. If string (non empty),
        it will be considered as the name of the only feature to be used
    :param drop_na: boolean. Ignore instances from the test set that have
        any NA/missing value, and assign them NaN as prediction. NA values are
        either NaN or None (and optionally +-Inf, see `inf_is_na`).
        Set this argument to False only if the classifier handles NaNs
    :param inf_is_na: boolean (default: True), tells if +-Inf have to be
        considered NA. This argument is ignored if `drop_na` is False
    :param inplace: boolean or string, default:False. If non-empty string or True,
        modify inplace the dataframe by using this argument as the name of
        the column to be added to the dataframe with the predicted scores/classes
        (if this argument is True, the column name defaults to 'prediction'), and
        return None.
        If False, return the predictions instead, in form of a numpy array

    :return: None or a numpy array of the predicted anomaly scores
    """
    if isinstance(features, str):
        features = [features]
    tst_set = testset if features is None else testset[features]
    _scores = None
    if drop_na:
        invalid = isna(tst_set, inf_is_na=inf_is_na)
        if invalid.any():
            # if we have some NA value, proceed to compute only others:
            _scores = np.full(len(tst_set), np.nan)
            valid = ~invalid
            _scores[valid] = prediction_function(clf, tst_set[valid].values.copy())

    # is _scores is None, either we do not need to drop na (the classifier
    # supports NA), or the testset has only finite values. In both cases
    # we still need to compute _scores, so:
    if _scores is None:
        _scores = prediction_function(clf, tst_set.values.copy())

    if inplace:
        colname = 'prediction' if inplace is True else str(inplace)
        testset[colname] = _scores
    else:
        return _scores


def compute_metrics(metrics, y_true, y_pred, sample_weight=None, *args, **kwargs):
    """
    Compute the given evaluation metrics

    :param metrics: Function or iterable of functions. All functions must have
        signature `(y_true, y_pred, ..., sample_weight=None, ...)` as in most
        sklearn functions in the `sklearn.metrics` module (for details, see
        https://scikit-learn.org/stable/modules/model_evaluation.html)
        and must return either a numeric scalar, or a dict of metric names
        (string) mapped to their numeric scalars (for functions returning a
        scalar, the function name will be used as metric name)
    :param y_true: array of numeric values denoting the true
        classes / scores / probabilities
    :param y_pred: array of numeric values denoting the predicted
        classes / scores/ probabilities
    :param sample_weight: array of None. The sample weights, the same length
        of `y_true` and `y_pred`. None (the default) means: no/same weight
    :param args: optional positional arguments to be passed to the metric functions
    :param kwargs: optional keyword arguments to be passed to the metric functions

    :return: a dict of all computed metric names, mapped to their value
    """
    # Handle NaNs, because some evaluation metrics do not handle it:
    finite = np.isfinite(np.asarray(y_pred))
    if not finite.all():
        y_pred = np.asarray(y_pred)[finite]
        y_true = np.asarray(y_true)[finite]
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)[finite]

    # If input arrays are empty (e.g. they had all NaNs, see above), some
    # metrics might fail. Return an empoty dict in case. Note that:
    # `pd.DataFrame([{'metric1': 1, 'metric2': 'g'}, {}])` produces:
    # ```
    #    metric1 metric2
    # 0      1.0       g
    # 1      NaN     NaN
    # ```
    # (and this works for us. See `run`)
    ret = {}
    if len(y_true) > 0:
        for metric_func in ([metrics] if callable(metrics) else metrics):
            val = metric_func(y_true, y_pred, sample_weight=sample_weight,
                              *args, **kwargs)
            if not isinstance(val, dict):
                val = {metric_func.__name__: val}
            ret.update(val)
    return ret
