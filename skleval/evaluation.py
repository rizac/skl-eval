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
# from math import isnan

from multiprocessing import Pool, cpu_count
from itertools import product, combinations, chain

# import numpy as np
import pandas as pd
# from pandas.core.indexes.range import RangeIndex
import click

from skleval.core import classifier, evaluate, evaluate_cv


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


class M:
    """Model table columns"""
    # (Enums are simply an overhead here)
    TR_SET = 'training_set'
    CLF = 'clf'
    GROUND_TRUTH = 'ground_truth_column',
    PARAM_PREFIX = 'param_'
    SAMPLE_WEIGHT = 'sample_weight_column',
    FEAT_PREFIX = 'feat_'
    DROP_NA = 'drop_na'
    INF_ISNA = 'inf_is_na'


class E:
    VAL_SET = 'validation_set'
    CV_SPLIT = 'cv_splitter'
    GROUND_TRUTH = 'ground_truth_column'
    METRIC_PREFIX = 'metric_'
    SAMPLE_WEIGHT = 'sample_weight_column'
    PREDICT_FUNC = 'prediction_function'


def run(clf, clf_parameters, trainingset, validationset, features,
        tr_set_ground_truth_col, tr_set_sample_weight_col,
        drop_na, inf_is_na, prediction_function, evaluation_metrics,
        val_set_ground_truth_col, val_set_sample_weight_col=None,
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

    # tr and val sets:
    tr_sets, val_sets, cv_splitters = process_input_datasets(trainingset,
                                                             validationset,
                                                             unique_features,
                                                             tr_set_ground_truth_col,
                                                             tr_set_sample_weight_col,
                                                             val_set_ground_truth_col,
                                                             val_set_sample_weight_col)

    # compute total iterations, create a function yielding the product of
    # all arguments which define the number of evaluations to be executed
    _all_validations = len(val_sets) + len(cv_splitters)
    total_iters = len(tr_sets) * _all_validations * feat_iterations * \
        len(parameters)

    if verbose:
        print(pd.DataFrame([
            ['Reading configuration file', '', ''],
            ['==========================', '', ''],
            ['training set(s):', str(len(tr_sets)), '×'],
            ['parameters combination(s):', str(len(parameters)), '×'],
            ['features combination(s):', str(feat_iterations), '×'],
            ['validation set(s) / cv splitter(s):', str(_all_validations), '='],
            ['----------------------------', '', ''],
            ['Total number of evaluations:', str(total_iters), '']
        ]).to_string(justify='right', index=False, header=False, na_rep=''))
        print()

    # these columns might be null or empty string. If null, then pandas
    # displays NaN, which might be misleading. Convert them to empty string:
    tr_set_ground_truth_col = tr_set_ground_truth_col or ''
    tr_set_sample_weight_col = tr_set_sample_weight_col or ''
    val_set_sample_weight_col = val_set_sample_weight_col or ''

    # Columns with a predefined finite sets of values can be set as
    # categorical data type, saving space. Let's create the Categorical dtypes
    # once for all now, and then cast each yielded dataframe.
    # PLEASE NOTE: None and NaN are not allowed in the categories, because it
    # is reserved for missing values (e.g.: categories=[1,2], dataframe has a
    # value=3 => the value will be None/NaN):

    # First start with the classifier parameters
    model_categorical_columns = {
        M.TR_SET: categorical_dtype(tr_sets.keys()),
        M.CLF: categorical_dtype([clf_path]),
        M.GROUND_TRUTH: categorical_dtype([tr_set_ground_truth_col]),
        M.SAMPLE_WEIGHT: categorical_dtype([tr_set_sample_weight_col]),
        **{M.PARAM_PREFIX + p: categorical_dtype(v) for p, v in clf_parameters.items()}
    }

    eval_categorical_columns = {
        E.VAL_SET: categorical_dtype(
            chain(val_sets.keys(), tr_sets.keys() if cv_splitters else [])
        ),
        E.GROUND_TRUTH: categorical_dtype([val_set_ground_truth_col]),
        E.SAMPLE_WEIGHT: categorical_dtype([val_set_sample_weight_col]),
        E.CV_SPLIT: categorical_dtype(cv_splitters.keys()),
        E.PREDICT_FUNC: categorical_dtype([prediction_function_path])
    }

    # Create the pool for multi processing (if enabled) and run the evaluations:
    pool = Pool(processes=int(cpu_count())) if multi_process else None
    # map function:
    _map = pool.imap_unordered if pool is not None else \
        lambda function, iterable, **kwargs: map(function, iterable)
    # map arguments:
    # clf_name, clf_class, clf_parameters, (tr_name, trainingset), \
    # tr_set_ground_truth_col, tr_set_sample_weight_col, \
    # prediction_function, prediction_function_name, \
    # val_sets, cv_splitters, features, unique_features, drop_na, \
    # inf_is_na, evaluation_metrics, val_set_ground_truth_col, \
    # val_set_sample_weight_col = args

    #         clf_name, clf_class, clf_parameters, (tr_name, trainingset), \
    #             tr_set_ground_truth_col, tr_set_sample_weight_col, \
    #             prediction_function, prediction_function_name, \
    #             val_sets, cv_splitters, features, unique_features, drop_na, \
    #             inf_is_na, evaluation_metrics, val_set_ground_truth_col, \
    #             val_set_sample_weight_col = args

    _args = product([clf_path], [clf_class], parameters, tr_sets.items(),
                    [tr_set_ground_truth_col], [tr_set_sample_weight_col],
                    [prediction_function], [prediction_function_path],
                    [val_sets], [cv_splitters], feat_iterator, [unique_features],
                    [drop_na], [inf_is_na], [evaluation_metrics],
                    [val_set_ground_truth_col], [val_set_sample_weight_col])

    models, evaluations, warnings = [], [], WarningContainer()
    progressbar = click.progressbar if verbose else _progressbar_mock
    with progressbar(length=total_iters, fill_char='o', empty_char='.',
                     file=sys.stderr, label='Computing') as pbar:
        chunksize = 100 if total_iters > 100 else 10 if total_iters > 10 else 1
        try:
            for id_, (model, evals, warns) in \
                    enumerate(_map(_evaluate_mp, _args, chunksize=chunksize), 1):
                pbar.update(len(evals))
                warnings.update(warns)
                models.append({'id': id_, **model})
                evaluations.extend({'model_id': id_, **e} for e in evals)

                if id_ % chunksize == 0:
                    models_df = _make_df(models, model_categorical_columns)
                    evaluations_df = _make_df(evaluations, eval_categorical_columns)
                    models, evaluations = [], []
                    yield models_df, evaluations_df

            if models:
                models_df = _make_df(models, model_categorical_columns)
                evaluations_df = _make_df(evaluations, eval_categorical_columns)
                yield models_df, evaluations_df

            if verbose and warnings:
                print('\nWarnings:')
                print("=========")
                print("(duplicated messages will be shown only once)")
                print("")
                print("\n".join(warnings.values()))

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


def process_input_datasets(training_set, validation_set, unique_features,
                           tr_set_ground_truth_col, tr_set_sample_weight_col,
                           val_set_ground_truth_col, val_set_sample_weight_col):
    # test sets (pandas DataFrames):
    _cols = list(unique_features) + [val_set_ground_truth_col]
    if val_set_sample_weight_col:
        _cols.append(val_set_sample_weight_col)
    _paths = [validation_set] if isinstance(validation_set, str) else validation_set
    v_sets, cv_splitters = {}, {}
    for pth in _paths:
        try:
            v_sets[pth] = read_hdf(pth, mandatory_columns=_cols)
        except Exception:  # noqa
            try:
                mod_sel_module = _load_pyobj("sklearn.model_selection")
                pth_ = pth.strip()
                mod_prefix = mod_sel_module.__name__ + '.'
                if pth_.startswith(mod_prefix):
                    pth_ = pth_[len(mod_prefix):]
                if pth_[-1:] != ')':
                    pth_ += '()'
                assert hasattr(mod_sel_module, pth_.rsplit('(', 1)[0])
                localz = {'_': mod_sel_module}
                exec('ret = _.%s' % pth_, localz)
                cv_splitter = localz['ret']
                assert 'BaseCrossValidator' in [_.__name__ for _ in cv_splitter.__class__.mro()]
                assert hasattr(cv_splitter, 'split') and callable(cv_splitter.split)
                cv_splitters[pth] = cv_splitter  # <- boolean "is_cv"
            except Exception:
                raise ValueError('"%s" is nor a valid path to a HDf file '
                                 '(validation set), neither a scikit-learn '
                                 'splitter (cross validation). '
                                 'Check typos' % pth)

    if not v_sets and not cv_splitters:
        raise ValueError('No validation set found (provide paths to HDf files or '
                         'scikit cross validator classes)')

    # training sets (pandas DataFrames):
    _cols = list(unique_features)
    if tr_set_ground_truth_col:
        _cols.append(tr_set_ground_truth_col)
    if tr_set_sample_weight_col:
        _cols.append(tr_set_sample_weight_col)
    _paths = [training_set] if isinstance(training_set, str) else training_set
    tr_sets = {f: read_hdf(f, mandatory_columns=_cols) for f in _paths}

    return tr_sets, v_sets, cv_splitters


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


def read_hdf(path_or_buf, *, key=None, columns=None, mandatory_columns=None,
             **kwargs):
    """Read the HDF from the given file path and return a pandas DataFrame.
    Wrapper around `pandas.read_hdf` with two features added:
    1. When `path_or_buf` is a path to file (`str`), The `key` argument (HDF
       group to read) can be specified as suffix after a double colon "::".
       E.g.: read_hdf('/myfile.hdf::table1', ...)`. If both the semicolon and
       the `key` arguments are specified, a `ValueError` is raised
    2. An additional keyword argument `mandatory_columns` is like the `columns`
       argument, but it requires also that the given columns exist. The function
       first builds the list of all columns (either mandatory or not), and then
       if `mandatory_columns` is given, raises `ValueError` if any given column
       is not in the loaded DataFrame
    """
    if isinstance(path_or_buf, str) and '::' in path_or_buf:
        if key is not None:
            raise ValueError('Can not read HDF: key "%s" specified in the path '
                             'with double colon but also given as argument')
        path_or_buf, key = path_or_buf.rsplit('::', 1)

    # merge columns and mandatory_columns, and read HDF:
    columns = list(columns or [])
    columns += [_ for _ in list(mandatory_columns or []) if _ not in columns]
    ret = pd.read_hdf(path_or_buf, key=key, columns=columns or None, **kwargs)
    # if mandatory_columns is given, check it:
    missing_columns = set(mandatory_columns or []) - set(ret.columns)
    if missing_columns:
        # format error and raise:
        err_msg = "Missing column(s) in HDF table: "
        err_msg += ', '.join('"%s"' % _ for _ in missing_columns)
        if isinstance(path_or_buf, str):
            err_msg += '(file: %s)' % path_or_buf
        raise ValueError(err_msg)
    return ret


def categorical_dtype(categories):
    """Create and return a pandas CategoricalDType from the given categories

    :param categories: iterable of Python objects. Anything
        for which `pandas.isna` returns True (e.g. NaN, None) will be skipped,
        as those values are reserved for values not in the categories:
        ```
        cat = categorical_dtype(['a', 1, None])  # <- None will be ignored
        pd.Series(data=['a', 1, 'this will be NaN'], dtype=cat)
        0      a
        1      1
        2    NaN
        dtype: category
        Categories (2, object): ['a', 1]
        ```
    """
    unique_categs, categs = set(), []
    for cat in categories:
        if cat in unique_categs or pd.isna(cat):
            # these 2 cases raise, so avoid
            continue
        unique_categs.add(cat)
        categs.append(cat)
    return pd.CategoricalDtype(categories=categs)  # (categories=[] is ok)


def _make_df(list_of_dicts, categorical_columns_dict):
    """Creats a dataframe from the given list of dicts

    :param categorical_columns_dict: a dict[str, pd.CategoricalDType] denoting
        columns whose type will be converted to categorical: if any column
        of the dataframe has string data spanning a finite set of possible
        values known a priori, you can initialize a pandas
        CategoricalDType`s with those values and pass it here. Categorical
        data type are much moire efficient in terms of memory usage
    """
    dfr = pd.DataFrame(list_of_dicts)
    for col, dtyp in categorical_columns_dict.items():
        dfr[col] = dfr[col].astype(dtyp, copy=True)
    return dfr


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
            tr_set_ground_truth_col, tr_set_sample_weight_col, \
            prediction_function, prediction_function_name, \
            val_sets, cv_splitters, features, unique_features, drop_na, \
            inf_is_na, evaluation_metrics, val_set_ground_truth_col, \
            val_set_sample_weight_col = args

        # Features might be a tuple. Dataframes interpret tuples not as multi
        # selector, but as single "scalar" key. Thus convert to list. Also,
        # let's convert string to a one element list, for safety
        if isinstance(features, str):
            features = [features]
        elif not isinstance(features, list):
            features = list(features)

        model = {
            M.CLF: clf_name,
            **{M.PARAM_PREFIX + k: v for k, v in clf_parameters.items()},
            M.TR_SET: tr_name,
            **{M.FEAT_PREFIX + f: f in features for f in unique_features},
            M.GROUND_TRUTH: tr_set_ground_truth_col,
            M.SAMPLE_WEIGHT: tr_set_sample_weight_col,
            M.DROP_NA: drop_na,
            M.INF_ISNA: inf_is_na,
        }

        base_eval_dict = {
            E.VAL_SET: None,
            E.CV_SPLIT: None,
            E.GROUND_TRUTH: val_set_ground_truth_col,
            E.SAMPLE_WEIGHT: val_set_sample_weight_col,
            E.PREDICT_FUNC: prediction_function_name,
        }

        # these parameters can be input as empty string to indicate None,
        # NOTE: they MUST be returned as string in the dicts above, but need
        # to be passed as None in the function calls below. So convert now:
        tr_set_ground_truth_col = tr_set_ground_truth_col or None
        tr_set_sample_weight_col = tr_set_sample_weight_col or None
        val_set_sample_weight_col = val_set_sample_weight_col or None

        clf = None
        if val_sets:
            clf = classifier(clf_class, clf_parameters, trainingset,
                             features=features, drop_na=drop_na,
                             inf_is_na=inf_is_na,
                             ground_truth=tr_set_ground_truth_col,
                             sample_weight=tr_set_sample_weight_col)

        evaluations = []
        for validation_name, validator in val_sets.items():
            eval_dict = dict(base_eval_dict)
            eval_dict[E.VAL_SET] = validation_name
            # validator is a validation/test set (pd DataFrame)
            y_pred, eval_ = evaluate(clf, validator, val_set_ground_truth_col,
                                     prediction_function, evaluation_metrics,
                                     sample_weight=val_set_sample_weight_col,
                                     features=features,
                                     drop_na=drop_na, inf_is_na=inf_is_na)
            eval_dict.update({E.METRIC_PREFIX + k: v for k, v in eval_.items()})
            evaluations.append(eval_dict)

        for validation_name, validator in cv_splitters.items():
            eval_dict = dict(base_eval_dict)
            eval_dict[E.VAL_SET] = tr_name
            eval_dict[E.CV_SPLIT] = validation_name
            # validator is a splitter class for cross-validation:
            y_pred, eval_ = evaluate_cv(clf_class, clf_parameters, trainingset,
                                        validator, val_set_ground_truth_col,
                                        prediction_function, evaluation_metrics,
                                        sample_weight_fit=tr_set_sample_weight_col,
                                        sample_weight_eval=val_set_sample_weight_col,
                                        features=features,
                                        drop_na=drop_na, inf_is_na=inf_is_na)

            eval_dict.update({E.METRIC_PREFIX + k: v for k, v in eval_.items()})
            evaluations.append(eval_dict)

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
