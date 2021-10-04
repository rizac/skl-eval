"""
Command line interface (CLI) of the evaluation routine

Created on 01 July 2021

@author: Riccardo Z (rizac@github.com)
"""
import os
import shutil
import sys
import yaml
import click
import pandas as pd


@click.command()
@click.argument("outputdir", metavar='[outputdir]')
def copy_example_files(outputdir):
    """
    Create an example file `evalconfig.yaml` with all necessary instructions to
    configure your own evaluation. Included also are three datasets required
    in the configuration (one training set and two validation sets in HDF format)
    and a Python module with user-defined prediction function and evaluation
    metrics, in order to show how to extend the default scikit methods and
    functions, if needed.

    [outputdir] the output directory. NOTE: the directory must not exist
        and will be created by the program
    """
    try:
        src = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        shutil.copytree(src, outputdir)
    except Exception as exc:
        print('Error: %s' % str(exc), file=sys.stderr)
        sys.exit(1)
    print('Files copied to %s' % outputdir)
    sys.exit(0)


@click.command()
@click.option('config_path', '-c', '--config',
    help=('configuration YAML file path. All paths therein will be '
          'relative to the file path, not the current working directory (cwd)'),
    required=True
)
@click.option('--single-process', '-sp', is_flag=True,
    help=('Run the program in a single process. By default, the program will '
          'attempt use multiple parallel subprocesses to run faster')
)
@click.option('--no-progress', '-np', is_flag=True,
    help=('Do not show any progress information. By default, a progress bar '
          'and the estimated time available are printed on the user terminal '
          '(standard error)')
)
@click.argument("outputfile", metavar='[outputfile]')
def run(config_path, single_process, no_progress, outputfile):
    """
    Perform machine learning evaluation by iteratively creating, training and
    testing models according to a configuration file (option -c), and storing
    evaluation results in the provided output file [outputfile].

    [outputfile] will be written in the easily accessible HDF format (see e.g.
    `pandas.read_hdf` in Python) and consists of two linked tables
    (`key` argument of `pandas.read_hdf`):

    \b
     - "models": where the information of each trained model is stored

    \b
     - "evaluations": where the evaluation metrics for each combination of
       (trained model, supplied test set) are stored

    \b
    (the program captures all warnings displaying them once on the standard
    output at the end of the whole process)
    """
    # For command line functions, importing inside the functions might speed
    # Fp whe we just import the module avoiding useless import
    from skleval.evaluation import run

    with open(config_path) as stream:
        cfg = yaml.safe_load(stream)
    cwd = os.getcwd()
    config_dir_path = os.path.abspath(os.path.dirname(config_path))
    os.chdir(config_dir_path)
    # Add the config dir to the system path (note: path must be absolute, it
    # seem not to work from the terminal otherwise):
    sys.path.append(config_dir_path)
    _prefix = ''
    try:
        _prefix = 'model'
        model = cfg[_prefix]

        _prefix = 'model.classifier'
        clf = model['classifier']
        clf_parameters = model['parameters']

        _prefix = 'model.training_set'
        trset = model['training_set']
        training_set = trset['file_path']
        tr_set_ground_truth_col = trset['ground_truth_column']
        tr_set_sample_weight_col = trset['sample_weight_column']
        features = trset['features']
        drop_na = trset['drop_na']
        inf_is_na = trset['inf_is_na']

        _prefix = 'evaluation'
        eval = cfg['evaluation']
        validation_set = eval['validation']
        val_set_ground_truth_col = eval['ground_truth_column']
        val_set_sample_weight_col = eval['sample_weight_column']
        prediction_function = eval['prediction_function']
        evalmetric_funcs = eval['metrics']

        # (clf, clf_parameters, trainingset, validationset, features,
        #  tr_set_ground_truth_col, tr_set_sample_weight_col,
        #  drop_na, inf_is_na, prediction_function, evaluation_metrics,
        #  val_set_ground_truth_col, val_set_sample_weight_col=None,
        #  multi_process=True, verbose=False):

        with pd.HDFStore(outputfile, 'w') as store:
            for model_df, eval_df in run(clf, clf_parameters, training_set,
                                         validation_set, features,
                                         tr_set_ground_truth_col,
                                         tr_set_sample_weight_col, drop_na,
                                         inf_is_na, prediction_function,
                                         evalmetric_funcs,
                                         val_set_ground_truth_col,
                                         val_set_sample_weight_col,
                                         multi_process=not single_process,
                                         verbose=not no_progress):
                store.append('models', model_df, format='table')
                store.append('evaluations', eval_df, format='table')

        final_str = "Evaluation completed, output written to: %s" % \
            os.path.abspath(outputfile)
        print("\n" + ('='*len(final_str)))
        print("\n" + final_str)

    except KeyError as kerr:
        print('Key "%s.%s" not found in "%s"' %
              (str(_prefix), str(kerr.args[0]), config_path),
              file=sys.stderr)
        sys.exit(1)
    finally:
        sys.path.pop()
        os.chdir(cwd)

    sys.exit(0)


if __name__ == '__main__':
    run()  # noqa
