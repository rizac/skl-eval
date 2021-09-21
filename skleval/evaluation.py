"""
Evaluation (classifiers creation and performance estimation) with pandas and sklearn
A set (training set, test set) is a pandas DataFrame where each row represents
an instance and each column a feature

Created on 1 Nov 2019

@author: riccardo
"""
import collections
import importlib
import inspect
import math
import sys
import warnings
from contextlib import contextmanager

from multiprocessing import Pool, cpu_count
from itertools import product, combinations

import numpy as np
import pandas as pd
# from pandas.core.indexes.range import RangeIndex
import click
# These are not used but are helpful IO function that we want to expose publicly:
from joblib import dump as save_clf, load as load_clf  # noqa


def _load_pyobj(py_path):
    """Import and return the given Python object from a path
    e.g.
    logloss_func = load_pyobj('sklearn.metrics.log_loss')
    """
    mod_path, att_name = py_path, None
    if '.' in py_path:
        mod_path, att_name = py_path.rsplit('.', 1)

    try:
        # try to see if we can load a module
        obj = importlib.import_module(mod_path)
    except ModuleNotFoundError as exc:
        # If we failed, it might be that mod_path is actually a class path:
        try:
            obj = _load_pyobj(mod_path)
        except Exception:
            raise exc from None

    try:
        return getattr(obj, att_name)
    except AttributeError:
        raise ValueError('"%s" not found in "%s"' % (att_name, mod_path))


def load_pyclass(py_path):
    obj = _load_pyobj(py_path)
    if not inspect.isclass(obj):
        raise ValueError('"%s" found but not a Python class' % py_path)
    return obj


def load_pyfunc(py_path):
    obj = _load_pyobj(py_path)
    if not inspect.isfunction(obj):
        raise ValueError('"%s" found but not a Python function' % py_path)
    return obj


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


def classifier(clf, clf_parameters, trainingset, features=None, drop_na=True,
               inf_is_na=True, **fit_kwargs):
    """
    Return a scikit learn model fitted with the data of `trainingset`.
    See :func:`save_clf` and :func:`load_clf` for serializing the created
    classifier

    :param clf: a class denoting a sklearn classifier (e.g.
        :class:`sklearn.ensemble.IsolationForest`) or its string path
        ("sklearn.ensemble.IsolationForest"): in this latter case, the class
        will be dynamically loaded from the path
    :param clf_parameters: the parameters of the classifier `__init__` method
    :param trainingset: pandas DataFrame denoting the training set
        (rows:instances, columns: features)
    :param features: features to train/fit the classifier with. It must be a
        list of columns of the training and test sets. If None (the default),
        all columns will be used. If string, it denotes the only column to be
        used
    :param drop_na: boolean. Drop instances from the training set that have
        any NA/missing value, before fitting the classifier. NA values are
        either NaN or None (and optionally +-Inf, see `inf_is_na`).
        Set this argument to False only if the classifier handles NaNs
    :param inf_is_na: boolean (default: True), tells if +-Inf have to be
        considered NA. This argument is ignored if `drop_na` is False
    :param fit_kwargs: optional keyword arguments for the classifier `fit` method

    :return: a fitted classifier object
    """
    clf_class = load_pyclass(clf) if isinstance(clf, str) else clf
    clf = clf_class(**clf_parameters)
    tr_set = trainingset if features is None else trainingset[tolst(features)]
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
    :param features: the column(s) of the dataframe denoting the features to
        be used, i.e. the features that `clf` has been trained with. If None
        (the default) all columns are used. If string (non empty), it will be
        considered as the name of the only feature to be used
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
    features = None if features is None else tolst(features)
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

    :param metrics: Function, string, or iterable of functions/strings.
        Any string must denote a Python path to a metric function that will
        be loaded dynamically. All functions must have signature
        `(y_true, y_pred, ...)` as in most sklearn functions (see module
        `sklearn.metrics`) and must return either a numeric scalar, or a dict
        of metric names (string) mapped to their numeric scalars (for functions
        returning a scalar, the function name will be used as metric name)
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
        for metric_func in yield_evauation_functions(metrics):
            val = metric_func(y_true, y_pred, sample_weight=sample_weight,
                              *args, **kwargs)
            if not isinstance(val, dict):
                val = {metric_func.__name__: val}
            ret.update(val)
    return ret


def yield_evauation_functions(evaluation_funcs):
    if not hasattr(evaluation_funcs, '__iter__') or \
            isinstance(evaluation_funcs, str):
        evaluation_funcs = [evaluation_funcs]

    for eval_func in evaluation_funcs:
        if isinstance(eval_func, str):
            yield load_pyfunc(eval_func)
        elif not callable(eval_func) or not hasattr(eval_func, '__name__'):
            raise ValueError("Not a evaluation metric function: %s"
                             % str(eval_func))
        else:
            yield eval_func


def hdf_nrows(filepath, key=None):
    """Get the number of rows of the given HDF.

    :param filepath: the HDF file path
    :param key: the key denoting the table in the HDF. If None, it will default
    to the only key found. A ValueError is raised if several keys are present
    and `key` is None, or if if key is not found in the HDF
    """
    store = pd.HDFStore(filepath)
    try:
        keys = list(store.keys())
        if key is None:
            if len(keys) == 1:
                key = keys[0]
            else:
                raise ValueError('Unable to get number of rows: '
                                 'HDF has more than 1 key')
        try:
            return store.get_storer(key).nrows
        except KeyError:
            raise ValueError('Key "%s" not found in HDF' % key)
    finally:
        store.close()


def feat_combinations(features):
    """Yields the total number of features combination

    :features: iterable of strings denoting the feature names. If string, it
        will return a list with the feature name as single element.
    """
    features = tolst(features)
    for k in range(len(features)):
        yield from combinations(features, k+1)


def feat_combinations_count(features):
    """Return the integer representing the total number of feature combinations
    from the given list of features"""
    n = len(features)
    return int(math.fsum(num_comb_nk(n, k+1) for k in range(n)))


def num_comb_nk(n, k):
    """Number of combinations of k elements taken from a set of n elements"""
    f = math.factorial
    return int(f(n) / f(k) / f(n-k))


_SAMPLE_WEIGHT_STR = 'sample_weight'


def run(clf, clf_parameters, trainingset, testset, features, ground_truth_column,
        drop_na, inf_is_na, prediction_function, evaluation_metrics,
        multi_process=True, verbose=False):
    """
    Run the evaluation from the given arguments. Yields two dicts, one with the
    trained classifier information, the second with the evaluation information

    :param clf: The classifier. Either a Python class denoting a sklearn classifier,
        or a string denoting the path to the classifier, e.g.
        'sklearn.ensemble.IsolationForest'
    :param clf_parameters: dict. The parameters of the classifier, as
        dict[str] - > list[values]
    :param trainingset: the path to the HDF file of the training set, or a list of
        paths. NOTE: if the HDF file contains several groups (directory-like
        structures) and you want to load a table from a specific group, append
        the group name to the end of the path after a double colon "::". The
        group name will be used as the `key` argument of :func:`pandas.read_hdf`.
        When no group is specified, it is expected that the HDF contains a
        single table so that the group name is not necessary (as per-pandas doc)
    :param testset: the path to the HDF file of the training set, or a list of
        paths. See note on `trainingset` for the concept of path (string or
        list of objects)
    :param features: the features, as iterable of strings
    :param ground_truth_column: string, the column with booleans or [0,1] values
        denoting the true class labels in each test set.
    :param drop_na: boolean, set to True if instances with any missing value
        (NA) should be removed from the training set and ignored in the test
        (i.e., the classifier algorithm does not support NaNs)
    :param inf_is_na: boolean, tells if +-inf are considered
        NA
    :param prediction_function: a string or a function denoting the score function
    :param evaluation_metrics: list of functions (Python callables) or
        string path to Python functions computing evaluation metrics.
        All functions must have signature `(y_true, y_pred, sample-weight=None, ...)`
        and return either a scalar, or a dict of metric names (str) mapped to
        their scalar value. See 'sklearn.metrics'
    :param multi_process: boolean (default: True) use parallel sub-processes
        to speed up execution
    :param verbose: if True (default: False): print progress bar to the screen

    :return: yields two dicts, one with the classifier information, the second
        with the evaluation information (including the evaluation metrics values)
    """
    # evaluation metrics:
    evaluation_metrics = process_evaluation_metrics(evaluation_metrics)

    # classifier and path:
    clf_path, clf_class = process_classifier(clf)

    # prediction function and path:
    prediction_function_path, prediction_function = \
        process_prediction_function(prediction_function, clf_class, clf_path)

    # clf parameters:
    parameters = process_parameters(clf_parameters)

    # features:
    unique_features, feat_iterator, feat_iterations = process_features(features)

    # training sets (pandas DataFrames):
    _cols = list(unique_features)
    trainingsets = {
        f: read_hdf(f, columns=_cols, mandatory_columns=_cols)
        for f in tolst(trainingset)
    }
    # test sets (pandas DataFrames):
    # (note: if a requested column is missing on the file, `read_hdf` works anyway)
    _cols += [ground_truth_column]
    testsets = {
        f: read_hdf(f, columns=_cols + [_SAMPLE_WEIGHT_STR], mandatory_columns=_cols)
        for f in tolst(testset)
    }

    # compute total iterations, create a function yielding the product of
    # all arguments which define the number of evaluations to be executed
    total_iters = len(trainingsets) * len(testsets) * feat_iterations * len(parameters)
    if verbose:
        print(pd.DataFrame([
            ['Reading configuration file', '', ''],
            ['==========================', '', ''],
            ['training set(s):', str(len(trainingsets)), '×'],
            ['parameters combination(s):', str(len(parameters)), '×'],
            ['features combination(s):', str(feat_iterations), '×'],
            ['test set(s):', str(len(testsets)), '='],
            ['----------------------------', '', ''],
            ['Total number of evaluations:', str(total_iters), '']
        ]).to_string(justify='right', index=False, header=False, na_rep=''))
        print()

    # clf_class, clf_parameters, trainingset, testsets, features, \
    # true_class_column, drop_na, eval_functions = args
    iterargs = product([clf_path], [clf_class], parameters,
                       trainingsets.items(), [prediction_function],
                       [prediction_function_path], [testsets], feat_iterator,
                       [unique_features], [ground_truth_column], [drop_na],
                       [inf_is_na], [evaluation_metrics])

    # Create the pool for multi processing (if enabled) and run the evaluations:
    pool = Pool(processes=int(cpu_count())) if multi_process else None
    progressbar = click.progressbar if verbose else _progressbar_mock

    with progressbar(length=total_iters, fill_char='o', empty_char='.',
                     file=sys.stderr, label='Computing') as pbar:
        mapfunc = pool.imap_unordered if pool is not None else \
            lambda function, iterable, *args, **kwargs: map(function, iterable)
        chunksize = max(1, int(total_iters/100))
        try:
            for id_, (model, evaluations, warnings) in \
                    enumerate(mapfunc(_evaluate_mp, iterargs, chunksize=chunksize), 1):
                # model['id'] = id_
                model_df = pd.DataFrame([model])
                model_df.insert(0, 'id', id_)  # insert id first
                evaluations_df = pd.DataFrame(evaluations)
                evaluations_df.insert(0, 'model_id', id_)  # insert model_id first
                # evaluations_df['model_id'] = id_  # assign on a dataframe, easier
                yield model_df, evaluations_df, warnings
                pbar.update(len(evaluations_df))
            # close and join the pool, even if for imap is not strictly necessary
            # it might prevent memory leaks (https://stackoverflow.com/q/38271547):
            if pool is not None:
                pool.close()
                pool.join()
        except Exception as exc:
            if pool is not None:
                _kill_pool(pool, str(exc))
            raise exc


def process_evaluation_metrics(evaluation_metrics):
    """
    Process the evaluation metrics returning a list of functions and checks
    the function signature (raiseing ValueError in case of errors)

    :param evaluation_metrics: iterable of evaluation metrics or string
        denoting evaluation metric functions with signature
        `(y_true, y_pred, ..., sample_weight=None, ...)`
    """
    evaluation_metrics = list(yield_evauation_functions(evaluation_metrics))
    for _ in evaluation_metrics:
        sig = inspect.signature(_)
        err_msg = ''
        if len(sig.parameters) < 3:
            err_msg = ('function has %d arguments, at least 3 required '
                       '(y_true, Y_pred, sample_weight=None)') \
                      % len(sig.parameters)
        if 'sample_weight' not in sig.parameters:
            err_msg = "missing function argument 'sample_weight' (optional)"
        swg = sig.parameters['sample_weight']
        if swg.default is not None:
            err_msg = "function argument 'sample_weight' must be optional " \
                      "with default None"
        if err_msg:
            raise ValueError('Error in evaluation metric %s: %s' % (_, err_msg))
    return evaluation_metrics


def process_classifier(clf):
    """
    Process the given classifier (str or sklearn class) and return the tuple:
    ```
    (clf_path:str, clf_class)
    ```
    :param clf: A Python class (scikit-learn class) or string denoting the
        Python path to such a class
    """
    # Preserve the classifier path as given in the input (if `clf` is `str`):
    # for instance, "sklearn.ensemble.IsolationForest" loads the `IsolationForest`
    # class whose path is "sklearn.ensemble._iforest.IsolationForest". Returning
    # the latter (private module) might be risky in case of sklearn refactoring
    # (which already happened)
    if isinstance(clf, str):
        clf_path = clf
        clf_class = load_pyclass(clf)
    else:
        clf_class = clf
        clf_path = clf_class.__module__ + '.' + clf_class.__name__
    return clf_path, clf_class


def process_prediction_function(prediction_function, clf_class, clf_path):
    """
    Process the given prediction function (str or function) and return
    the tuple:
    ```
    (prediction_function_path:str, prediction_function)
    ```
    :param prediction_function: the prediction function input
    :param clf_class: the classifier Python class
    :param clf_path: the classifier Python path (usually
        `clf_class.__module__` + `clf_class.__name__`)
    """
    if not callable(prediction_function):
        _ = getattr(clf_class, str(prediction_function), prediction_function)
        if callable(_):  # `prediction_function` arg is a class method
            prediction_function_path = clf_path + '.' + prediction_function
            prediction_function = _
        else:   # `prediction_function` arg is a Python path
            prediction_function_path = str(prediction_function)
            prediction_function = load_pyfunc(prediction_function_path)
    else:
        prediction_function_path = prediction_function.__module__ + \
                                   '.' + prediction_function.__name__
    return prediction_function_path, prediction_function


def process_parameters(clf_parameters):
    """
    Process the given classifier parameters, returning all possible
    combinations of their values in a `tuple[dict[str, val]]`

    Example:
    ```
    >>> process_parameters({"A": [0, 1], "B": "string"})
       ({"A":0, "B":"string"}, {"A":1, "B":"string"})

    :param clf_parameters: dict. The parameters of the classifier, as
        dict[str] - > list[values]
    """
    __p = []
    for pname, vals in clf_parameters.items():
        if not isinstance(vals, (list, tuple)):
            raise TypeError(("'%s' must be mapped to a list or tuple of "
                             "values, even when there is only one value "
                             "to iterate over") % str(pname))
        __p.append(tuple((pname, v) for v in vals))
    return tuple(dict(_) for _ in product(*__p))


def process_features(features):
    """
    Process the given features list and return the tuple:
    ```
    (unique_features:list, features_iterator:iterator, features_iterations:int)
    ```
    :param features: A list of features as read from YAML config: it can be
         a List[str] (-> compute and return all combinations) or a
         List[List[str]] (simply return the list)
    """
    unique_features = []  # set would be better, but does not preserve order
    if all(isinstance(_, (list, tuple)) for _ in features):
        features_iterator = features
        features_iterations = len(features)
        for feats in features:
            for feat in feats:
                if not isinstance(feat, str):
                    raise ValueError('features must be an Iterable[str] '
                                     'or Iterable[Iterable[str]]')
                if feat not in unique_features:
                    unique_features.append(feat)
    else:
        if not all(isinstance(_, str) for _ in features):
            raise ValueError('features must be an Iterable[str] '
                             'or Iterable[Iterable[str]]')
        for feat in features:
            if feat not in unique_features:
                unique_features.append(feat)
        features_iterator = feat_combinations(unique_features)
        features_iterations = feat_combinations_count(unique_features)
    return unique_features, features_iterator, features_iterations


def read_hdf(path_or_buf, *args, mandatory_columns=None, **kwargs):
    """Read the HDF from the given file path and return a pandas DataFrame.
    Wrapper around `pandas.read_hdf` with two features added:
    1. The `key` argument (HDF group to read) can be specified as suffix after
       a double colon "::" in `path_or_buf`, when the latter is a `str` (path
       to file). E.g.: read_hdf('/path/to/file.hdf::data', *args, **kwargs)`
    2. An additional keyword argument `mandatory_columns` checks that the
       given column(s) exist on the dataframe. Use this in conjunction with the
       `columns` argument, which loads only specific columns but does not check
       nor raises if any column is not found on the file
    In both cases, a ValueError is raised
    """
    if isinstance(path_or_buf, str):
        idx = path_or_buf.rfind('::')
        if idx > -1:
            key = path_or_buf[idx+2:]
            path_or_buf = path_or_buf[:idx]
            if len(args) or 'key' in kwargs:
                raise ValueError('Can not read HDF: key "%s" specified in the path '
                                 'with double colon but also given as argument')
            kwargs['key'] = key
    ret = pd.read_hdf(path_or_buf, *args, **kwargs)
    if mandatory_columns is not None:
        missing_columns = set(mandatory_columns) - set(ret.columns)
        if missing_columns:
            # format error and raise:
            err_msg = "Missing column(s) in HDF table: "
            err_msg += ', '.join('"%s"' % _ for _ in missing_columns)
            if isinstance(path_or_buf, str):
                err_msg +=  '(file: %s)' % path_or_buf
            raise ValueError(err_msg)
    return ret


@contextmanager
def _progressbar_mock(*a, **v):  # noqa

    class ProgressBarMock:
        def update(self, *args, **kwargs):
            pass
    yield ProgressBarMock()


def _evaluate_mp(args):
    """Wrapper function for calling evaluate_classifier inside pool.imap"""
    oldwarn = warnings.showwarning
    warnings.showwarning = WarningContainer()

    try:
        clf_name, clf_class, clf_parameters, (tr_name, trainingset), \
            prediction_function, prediction_function_name, \
            testsets, features, unique_features, true_class_column, drop_na, \
            inf_is_na, evaluation_metrics = args

        clf = classifier(clf_class, clf_parameters, trainingset, features=features,
                         drop_na=drop_na, inf_is_na=inf_is_na)

        # Model column(s):
        ##################
        # id:int
        # clf:str
        # param_<name>:val
        # ...
        # param_<name>:val
        # drop_na:bool
        # training_set:str
        # feat_<name>:bool
        # ...
        # feat_<name>:bool

        # Evaluation columns
        ####################
        # model_id:int
        # test_set:str
        # ground_truth_column:str
        # prediction_function:str
        # <evalmetric_name>:float
        # ...
        # <evalmetric_name>:float

        model = {
            'clf': clf_name,
            **{'param_%s' % k: v for k, v in clf_parameters.items()},
            'trainingset': tr_name,
            **{'feat_%s' % f: f in features for f in unique_features},
            'drop_na': drop_na, 'inf_is_na': inf_is_na
        }

        evaluations = []
        for testset_name, testset in testsets.items():
            sample_weight = _SAMPLE_WEIGHT_STR \
                if _SAMPLE_WEIGHT_STR in testset.columns else None
            y_pred, eval = evaluate(clf, testset, true_class_column,
                                    prediction_function, evaluation_metrics,
                                    sample_weight=sample_weight, features=features,
                                    drop_na=drop_na, inf_is_na=inf_is_na)
            eval_new = {
                'testset': testset_name,
                'true_class_column': true_class_column,
                'prediction_function': prediction_function_name,
                **{'evalmetric_%s' % k: v for k, v in eval.items()}
            }
            evaluations.append(eval_new)

        return model, evaluations, warnings.showwarning
    finally:
        warnings.showwarning = oldwarn


def tolst(obj):
    """Convert obj to a list of elements: if string, return `[obj]`, if
    list, return `obj`, if iterable, return `list(obj)`.
    """
    # IMPORTANT: the returned value must be a list, because it is used also in
    # pandas DataFrames column selector: in this case, a list is needed
    # (e.g., a tuple is interpreted as single key)
    if isinstance(obj, str):
        return [obj]
    if hasattr(obj, '__iter__'):
        return obj if isinstance(obj, list) else list(obj)
    raise ValueError('Expected iterable or string, found %s' % obj.__class__.__name__)


def _kill_pool(pool, err_msg):
    print('ERROR:')
    print(err_msg)
    try:
        pool.terminate()
    except ValueError:  # ignore ValueError('pool not running')
        pass


class WarningContainer(dict if sys.version_info[:2] >= (3, 6) else
                       collections.OrderedDict):
    """
    Ordered `dict` which captures warnings internally and handles multi-processing.

    Rationale: with Python multi-processing, capturing warnings
    (https://docs.python.org/3/library/warnings.html#warnings.simplefilter)
    works only within each sub-process. This results in several duplicated
    warnings being shown anyway on the terminal, regardless of the filter used
    with `simplefilter`.

    To fix this, an object of this class can be set **in each sub-process**
    as the default `warnings` writing function (this object is callable):
    ```
    warnings.showwarning = WarningContainer()`
    ```
    and then be yielded or returned to the main parent process, where it can
    be merged with all other WarningContainer `dict`s to produce a final
    `WarningContainer` with all messages from all sub-processes stored
    once. The messages can then be handled as usual, e.g. printing them to
    the standard output with no redundancies
    """
    def __call__(self, message, category, filename, lineno, file=None, line=None):
        """This function is called by the warnings module when a warning is issued
        For info see https://docs.python.org/3/library/warnings.html#warnings.showwarning
        """
        if str(message) in self:
            return

        self[str(message)] = warnings.formatwarning(message, category,
                                                    filename, lineno)


# def save_df(dataframe, filepath, **kwargs):
#     """Wrapper around `dataframe.to_hdf`.
#     Save the given dataframe as HDF file under `filepath`.
#
#     :param kwargs: additional arguments to be passed to pandas `to_hdf`,
#         EXCEPT 'format' and 'mode' that are set inside this function
#     """
#     if 'key' not in kwargs:
#         key = splitext(basename(filepath))[0]
#         if not re.match('^[a-zA-Z_][a-zA-Z0-9_]*$', key):
#             raise ValueError('Invalid file basename. '
#                              'Change the name or provide a `key` argument '
#                              'to the save_df function or change file name')
#         kwargs['key'] = key
#     kwargs.setdefault('append', False)
#     dataframe.to_hdf(
#         filepath,
#         format='table',
#         mode='w' if not kwargs['append'] else 'a',
#         **kwargs
#     )