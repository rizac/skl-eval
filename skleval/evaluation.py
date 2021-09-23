"""
Evaluation routine module

Created on 1 Nov 2019

@author: Riccardo Z <rizac@github.com>
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

# import numpy as np
import pandas as pd
# from pandas.core.indexes.range import RangeIndex
import click

from skleval.core import classifier, evaluate


def load_pyclass(py_path):
    """Loads a Python class from the specified path. E.g.
    `load_pyclass('sklearn.ensemble.IsolationForest')`

    :raise:
        `ModuleNotFoundError` (invalid path: module not found),
        `AttributeError` (invalid path: object not found in module)
        `ValueError` (valid path but loaded object is not a Python class)
    """
    obj = _load_pyobj(py_path)
    if not inspect.isclass(obj):
        raise ValueError('"%s" found but not a Python class' % py_path)
    return obj


def load_pyfunc(py_path):
    """Loads a Python class from the specified path. E.g.
    `load_pyclass('sklearn.metrics.log_loss')`

    :raise:
        `ModuleNotFoundError` (invalid path: module not found),
        `AttributeError` (invalid path: object not found in module)
        `ValueError` (valid path but loaded object is not a Python function)
    """
    obj = _load_pyobj(py_path)
    if not inspect.isfunction(obj):
        raise ValueError('"%s" found but not a Python function' % py_path)
    return obj


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

    return getattr(obj, att_name)


_SAMPLE_WEIGHT_STR = 'sample_weight'


def run(clf, clf_parameters, trainingset, validationset, features, ground_truth_column,
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
    :param validationset: the path to the HDF file of the validation set, or a
        list of paths. See note on `trainingset` for the concept of path
        (string or list of objects)
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
    _paths = [trainingset] if isinstance(trainingset, str) else trainingset
    trainingsets = {
        f: read_hdf(f, columns=_cols, mandatory_columns=_cols) for f in _paths
    }
    # test sets (pandas DataFrames):
    # (note: if a requested column is missing on the file, `read_hdf` works anyway)
    _cols += [ground_truth_column]
    _paths = [validationset] if isinstance(validationset, str) else validationset
    validationsets = {
        f: read_hdf(f, columns=_cols + [_SAMPLE_WEIGHT_STR], mandatory_columns=_cols)
        for f in _paths
    }

    # compute total iterations, create a function yielding the product of
    # all arguments which define the number of evaluations to be executed
    total_iters = len(trainingsets) * len(validationsets) * feat_iterations * len(parameters)
    if verbose:
        print(pd.DataFrame([
            ['Reading configuration file', '', ''],
            ['==========================', '', ''],
            ['training set(s):', str(len(trainingsets)), '×'],
            ['parameters combination(s):', str(len(parameters)), '×'],
            ['features combination(s):', str(feat_iterations), '×'],
            ['test set(s):', str(len(validationsets)), '='],
            ['----------------------------', '', ''],
            ['Total number of evaluations:', str(total_iters), '']
        ]).to_string(justify='right', index=False, header=False, na_rep=''))
        print()

    # clf_class, clf_parameters, trainingset, testsets, features, \
    # true_class_column, drop_na, eval_functions = args
    iterargs = product([clf_path], [clf_class], parameters,
                       trainingsets.items(), [prediction_function],
                       [prediction_function_path], [validationsets],
                       feat_iterator, [unique_features], [ground_truth_column],
                       [drop_na], [inf_is_na], [evaluation_metrics])

    # String columns can be saved as categorical, saving space. But we need to
    # create Categorical dtypes once for all to include all categories:
    model_categorical_columns = {
        'training_set': pd.CategoricalDtype(categories=trainingsets.keys()),
        'clf': pd.CategoricalDtype(categories=[clf_path])
    }
    # clf parameters whose values are all strings can also be categorical:
    for pname, pvals in clf_parameters.items():
        if all(isinstance(_, str) for _ in pvals):
            model_categorical_columns['param_%s' % pname] = \
                pd.CategoricalDtype(categories=pvals)
    eval_categorical_columns = {
        'validation_set': pd.CategoricalDtype(categories=validationsets.keys()),
        'true_class_column': pd.CategoricalDtype(categories=[ground_truth_column]),
        'prediction_function': pd.CategoricalDtype(categories=[prediction_function_path])
    }

    # Create the pool for multi processing (if enabled) and run the evaluations:
    pool = Pool(processes=int(cpu_count())) if multi_process else None
    progressbar = click.progressbar if verbose else _progressbar_mock

    with progressbar(length=total_iters, fill_char='o', empty_char='.',
                     file=sys.stderr, label='Computing') as pbar:
        mapfunc = pool.imap_unordered if pool is not None else \
            lambda function, iterable, *args, **kwargs: map(function, iterable)
        chunksize = 100 if total_iters > 100 else 1  # max(1, int(total_iters/100))
        try:
            for id_, (model, evaluations, warnings) in \
                    enumerate(mapfunc(_evaluate_mp, iterargs, chunksize=chunksize), 1):
                # model['id'] = id_
                model_df = pd.DataFrame([model])
                model_df.insert(0, 'id', id_)  # insert id first
                for col, dtype in model_categorical_columns.items():
                    model_df[col] = model_df[col].astype(dtype, copy=True)
                evaluations_df = pd.DataFrame(evaluations)
                evaluations_df.insert(0, 'model_id', id_)  # insert model_id first
                for col, dtype in eval_categorical_columns.items():
                    evaluations_df[col] = evaluations_df[col].astype(dtype, copy=True)
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


def feat_combinations(features):
    """Yields the total number of features combination

    :features: iterable of strings denoting the feature names. If string, it
        will return a list with the feature name as single element.
    """
    features = list(features) if not isinstance(features, list) else features
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
        # Note below: the variable name 'testsets' is legacy code, they are
        # actually validation sets
        clf_name, clf_class, clf_parameters, (tr_name, trainingset), \
            prediction_function, prediction_function_name, \
            testsets, features, unique_features, true_class_column, drop_na, \
            inf_is_na, evaluation_metrics = args

        # Features might be a tuple. Dataframes interpret tuples not as multi
        # selector, but as single "scalar" key. Thus convert to list. Also,
        # let's convert string to a one element list, for safety
        if isinstance(features, str):
            features = [features]
        elif not isinstance(features, list):
            features = list(features)

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
            'training_set': tr_name,
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
                'validation_set': testset_name,
                'true_class_column': true_class_column,
                'prediction_function': prediction_function_name,
                **{'evalmetric_%s' % k: v for k, v in eval.items()}
            }
            evaluations.append(eval_new)

        return model, evaluations, warnings.showwarning
    finally:
        warnings.showwarning = oldwarn


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
    warnings.showwarning = WarningContainer()
    ```
    and then be yielded or returned to the main parent process, where it can
    be merged with all other WarningContainer `dict`s to produce a final
    `WarningContainer` with all messages from all sub-processes stored
    once. The messages can then be handled as usual, e.g. printing them to
    the standard output with no redundancies
    """
    def __call__(self, message, category, filename, lineno, file=None, line=None):
        """
        This function is called by the warnings module when a warning is
        issued. For info see
        https://docs.python.org/3/library/warnings.html#warnings.showwarning
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
