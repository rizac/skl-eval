import sys
import unittest
import os
from itertools import product
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml
from click.testing import CliRunner
from sklearn.ensemble import IsolationForest

from skleval.evaluation import isna, dropna, feat_combinations, \
    feat_combinations_count, process_parameters, process_features, \
    process_prediction_function, read_hdf
from skleval.cli import run


def _decision_function(clf, X):
    """Mockup used in the tests below"""
    return clf.decision_function(X)


class Test(unittest.TestCase):

    def test_drop_na(self):
        dfr = pd.DataFrame({
            'f1': [1, np.nan, 4.3, 4, 5],
            'f2': [2.2, 4, np.inf, 3, 1.1],
            'f3': [2.2, 4, 3, -np.inf, 1.1],
            'f4': ['1', '2', 'a', 't', None]
        })
        self.assertEqual(isna(dfr).sum(), 2)  # first and last row
        pd.testing.assert_frame_equal(dropna(dfr), dfr.dropna(how='any'))

        self.assertEqual(isna(dfr, inf_is_na=True).sum(), 4)  # first and last row
        with pd.option_context('mode.use_inf_as_na', True):
            _dfr = dfr.dropna(how='any')
        pd.testing.assert_frame_equal(dropna(dfr, inf_is_na=True), _dfr)

    def test_feat_comb(self):
        feats_expected = {
            ('a',): [('a',)],
            ('a', 'b'): [('a',), ('a', 'b'), ('b',)],
            ('a', 'b', 'c'): [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'c'), ('b',), ('b', 'c'), ('c',)]
        }
        for feats, expected in feats_expected.items():
            self.assertEqual(expected, sorted(feat_combinations(feats)))
            self.assertEqual(len(list(feat_combinations(feats))), len(expected))

    @patch('sklearn.ensemble.IsolationForest.score_samples')
    def test_prediction_function(self, iforest_score_samples):
        # Note that IsolationForest.score_samples is called by decision_function
        # Note also that we could mock the latter but then it would not be
        # a function anymore (instead, a MagicMock) and thus some test will fail
        clf = IsolationForest
        clf_path = 'sklearn.ensemble.IsolationForest'
        paths,  funcs = [], []
        pred_func_path, pred_func = \
            process_prediction_function('decision_function', clf, clf_path)
        paths.append(pred_func_path)
        funcs.append(pred_func)

        pred_func_path, pred_func = \
            process_prediction_function('sklearn.ensemble.IsolationForest.decision_function', clf, clf_path)
        paths.append(pred_func_path)
        funcs.append(pred_func)

        try:
            sys.path.append(os.path.dirname(__file__))
            pred_func_path, pred_func = \
                process_prediction_function('test_eval._decision_function', clf, clf_path)
            paths.append(pred_func_path)
            funcs.append(pred_func)
        finally:
            sys.path.pop()

        pred_func_path, pred_func = \
            process_prediction_function(_decision_function, clf, clf_path)
        paths.append(pred_func_path)
        funcs.append(pred_func)

        # check that all functions are actually calling our decision_function:
        iforest = IsolationForest()
        iforest.fit([[0], [1]])
        self.assertEqual(iforest_score_samples.call_count, 0)
        for call_count, func in enumerate(funcs, 1):
            func(iforest, 9)  # dont need args, it's the mocked function
            self.assertEqual(iforest_score_samples.call_count, call_count)

    def test_read_hdf(self):
        testdata_dir = os.path.join(os.path.dirname(__file__), 'data')
        input_hdf = os.path.join(testdata_dir, 'testset_bad.hdf')
        # test with the key appended as '::'+ key:
        pd.testing.assert_frame_equal(read_hdf(input_hdf),
                                      read_hdf(input_hdf+'::segments'))
        # test by providing additional coplumns not existing ('bla'):
        pd.testing.assert_frame_equal(read_hdf(input_hdf),
                                      read_hdf(input_hdf, columns=['PGA', 'PGV',
                                                                   'outlier', 'bla']))
        # test errors

        # wrong key:
        with self.assertRaises(KeyError) as verr:
            read_hdf(input_hdf+'::abc')

        # key supplied twice:
        with self.assertRaises(ValueError) as verr:
            read_hdf(input_hdf+'::abc', 'key')

        # wrong key:
        with self.assertRaises(ValueError) as verr:
            read_hdf(input_hdf, columns=['PGA', 'PGV', 'outlier', 'bla'],
                     mandatory_columns=['PGA', 'PGV','outlier', 'bla'])

    def test_run(self):
        testdata_dir = os.path.join(os.path.dirname(__file__), 'data')
        destfile = os.path.join(testdata_dir, 'eval_results.hdf')
        for input_filename in ['evalconfig.yaml', 'evalconfig2.yaml']:
            inputconfig = os.path.join(testdata_dir, input_filename)
            try:
                runner = CliRunner()
                result = runner.invoke(run, ['-c', inputconfig, '--single-process',
                                             destfile])
                self.assertEqual(result.exit_code, 0)
                self.assertTrue(os.path.isfile(destfile))

                models_df = pd.read_hdf(destfile, key='models')
                # check models?
                with open(inputconfig) as stream:
                    cfg = yaml.safe_load(stream)
                feat_count = process_features(cfg['features'])[-1]
                params_count = len(process_parameters(cfg['classifier']['parameters']))
                trset_count = 1 if isinstance(cfg['trainingset'], str) else len(cfg['trainingset'])
                total_rows = int(feat_count * params_count * trset_count)
                self.assertEqual(len(models_df), total_rows)

                # check evaluations:
                eval_df = pd.read_hdf(destfile, key='evaluations')
                tsset_count = 1 if isinstance(cfg['testset'], str) else len(cfg['testset'])
                total_rows = len(models_df) * tsset_count
                self.assertEqual(len(eval_df), total_rows)
                for _, dfr in eval_df.groupby('model_id'):
                    # there are two rows in `dfr`, one relative to a "good" testset,
                    # where data has the "correct" label, and a "bad" testset, where
                    # the labels have been switched on purpose:
                    series_good = dfr.loc[dfr.testset.str.contains('_good')]
                    self.assertTrue(len(series_good) == 1)
                    series_good = series_good.iloc[0]
                    series_bad = dfr.loc[dfr.testset.str.contains('_bad')]
                    self.assertTrue(len(series_bad) == 1)
                    series_bad = series_bad.iloc[0]
                    # evm: the higher, the better, evm_loss: the lower, the better
                    evm = ['evalmetric_%s' % _ for _ in
                                ['average_precision_score',
                                 'pr_curve_best_threshold_f1score']]
                    evm = [c for c in dfr.columns if c in ['evalmetric_average_precision_score',
                                                           'evalmetric_pr_curve_best_threshold_f1score']]
                    evm_loss = [c for c in dfr.columns if '_error' in c or '_loss' in c]
                    # check evaluation values are consistent in the two testsets
                    # (good vs bad):
                    self.assertTrue((series_good[evm] > series_bad[evm]).all())
                    self.assertTrue((series_good[evm_loss] < series_bad[evm_loss]).all())

                # open the file and check
            finally:
                if os.path.isfile(destfile):
                    os.remove(destfile)