"""
Command line interface (CLI) of the evaluation routine

Created on 01 July 2021

@author: Riccardo Z (rizac@github)
"""
import os
import sys
import yaml
import click
import pandas as pd


import skleval.evaluation as evaluation


@click.argument("outputdir", metavar='[outputdir]')
def copy_example_files(outputdir):
    """
    Create an example file `evalconfig.yaml` with all necessary instructions to
    configure your own evaluation. Included also are three datasets required
    in the configuration (one training set and two test sets in HDF format)
    and a Python module with user-defined prediction function and evaluation
    metrics, in order to show how to extend the default scikit methods and
    functions, if needed.

    [outputdir] the output directory (if non existing, it will be created)
    """
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir)
    if not os.path.isdir(outputdir):
        print("%s does not exist and could not be created" % outputdir)
        sys.exit(1)
    files = ['evalconfig', 'evalutilities.py']
    for


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
    with open(config_path) as stream:
        cfg = yaml.safe_load(stream)
    cwd = os.getcwd()
    config_dir_path = os.path.abspath(os.path.dirname(config_path))
    os.chdir(config_dir_path)
    # Add the config dir to the system path (note: path must be absolute, it
    # seem not to work from the terminal otherwise):
    sys.path.append(config_dir_path)
    warnings = evaluation.WarningContainer()
    try:
        clf = cfg['classifier']['classname']
        prediction_function = cfg['prediction_function']
        clf_parameters = cfg['classifier']['parameters']
        trainingset = cfg['trainingset']
        testset = cfg['testset']
        features = cfg['features']
        ground_truth_column = cfg['ground_truth_column']
        drop_na = cfg['drop_na']
        inf_is_na = cfg['inf_is_na']
        evalmetric_funcs = cfg['evaluation_metrics']
        # destfile = cfg['evaluation_destination_file']
        with pd.HDFStore(outputfile, 'w') as store:
            try:
                for model_df, eval_df, wrngs in \
                        evaluation.run(clf, clf_parameters, trainingset,
                                       testset, features, ground_truth_column,
                                       drop_na, inf_is_na, prediction_function,
                                       evalmetric_funcs,
                                       multi_process=not single_process,
                                       verbose=not no_progress):
                    store.append('models', model_df, format='table')
                    store.append('evaluations', eval_df, format='table')
                    warnings.update(wrngs)
            finally:
                store.close()
        print("\nEvaluation performed, output written to: %s" %
              os.path.abspath(outputfile))

        if warnings:
            print('\nWarnings:')
            print("=========")
            print("(duplicated messages will be shown only once)")
            print("")
            print("\n".join(warnings.values()))

    except KeyError as kerr:
        print('Key "%s" not found in "%s"' % (str(kerr), config_path),
              file=sys.stderr)
        sys.exit(1)
    finally:
        sys.path.pop()
        os.chdir(cwd)

    sys.exit(0)


if __name__ == '__main__':
    run()  # noqa
