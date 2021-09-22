# skl-eval
Program to run iteratively scikit learn evaluations from pre-built
training and validation set(s): implement your own features, parameters 
and evaluation metrics in a configuration file, and let the program run 
iteratively any given combination of them. 

<!--
user@mymachine:~/mydir$ (PYTHONPATH=. ./.env/bin/python ./sdaas_eval/evaluate.py -c ./sdaas_eval/eval_configs/evalconfig.yaml /home/me/sdaas_eval_data/accelerometers2.eval.hdf)
-->

```console
$ skl-eval -c ./path/to/my-config.yaml /path/to/my-evaluation-result.hdf
   Reading configuration file
   ==========================
             training set(s):       1  ×
   parameters combination(s):      27  ×
     features combination(s):    4095  ×
                 test set(s):       4  =
 ----------------------------
 Total number of evaluations:  442260

Computing  [oooooooooooooooooooooooooooo........]   80%  04:43:12
```
(extract from the terminal during evaluation)

This program is particularly useful: 
 1. In those cases where a separate training set(s) and test set(s) are 
    available, e.g. when simpler evaluation by partitioning 
    (e.g. K-fold split) would be insufficient (if this is not the case,
    consider reading the [scikit evaluation page](https://scikit-learn.org/stable/modules/model_evaluation.html)
    as your case might be already covered there)
 2. To avoid the unnecessary burden of running all models in a loop
    within your Python code, and nicely show progressbar and estimated
    time available on the terminal
 3. To save your evaluations in a portable and simple tabular format
    (HDF file) whose structure is described at the bottom of the page


## Installation

Create a directory where to clone the project, `cd` in it and then
as usual:

```console
$ git clone <repository_url> .
$ python3 -m venv <env_dir>  # create virtual Python environment
$ source <env_dir>/bin/activate  # activate it (if not already activated)
$ pip install -e .  $ install the program
```
(-e is optional  and makes this package editable, meaning that any new 
`git pull` automatically updates the package without re-installation needed)

Note: **The virtual environment must be activated any time you want to use 
the program** (type `deactivate` on the terminal to de activate it)


### Test:

If you cloned the repository for developing new features:

```console
python -m unittest -f
```

## Usage

### Prepare your datasets

The datasets can be created as `DataFrame` objects with the `pandas` library
(installed with the  program) and then saved as HDF via the `DataFrame.to_hdf`
method

To run the program you need:

- One or more training set, with features arranged by column(s), and 
  instances arranged by rows
- One or more test sets, with the same features as the training set(s) and
  an additional "ground truth" column (which can be named as you like) 
  with the real values, e.g. class labels,  numeric scores. 
  Additionally, a column named 'sample_weight' can be added: if present, 
  it will be used in the `sample_weight` argument of most [evaluation
  metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
  

### Configure the evaluation(s)
Create the example configuration file via the init command:
```console
skl-eval-init <output_directory>
```
the file contains all documentation needed and can thus
be renamed and/or edited to configure your evaluation:

#### Setup dataset paths

By default, in the created config the training set and test set paths
refer to three example datasets also provided in the output directory.
The first thing to do then is to replace the paths with the paths of your 
datasets.


### Setup parameters and features

Then, you can configure the classifier type, its parameters,
the features to be used, and the evaluation metrics to compute. 
The program will then take care of building all combinations of 
parameters, features and metrics to calculate


### (Optional) Configure custom prediction functions and evaluation metrics
 
By default, scikit learn models implement their decision / prediction
functions via class methods (e.g. `predict`, `decision_function`), and
the package offers all necessary [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html). 
In both cases, you can just type the names of the functions you want to use
in the configuration file (e.g. "sklearn.metrics.log_loss").

However, in some cases you might want to use custom functions and metrics: 
the created configuration illustrates how to proceed as some paths refer
to user-defined functions implemented in an associated custom Python, which
is also copied in the same directory of the config file: because the evaluation 
config directory is added to the Python path at runtime, any Python module
implemented therein will be accessible.


### Run the evaluation

Once setup, you can run the evaluation to a dedicated output file in HDF format
(You can always type `skl-eval --help` for details)
```console
skl-eval -c <config_file> <output_hdf_file>
```

## Evaluation output file description

(this section is available also at the bottom of the configuration file)

<!-- DEVELOPERS NOTE: when modifying the Evaluation output file description,
it is recommended to modify it here and then copy/paste this section at the
bottom of data/evalconfig.yaml -->

The evaluation result file produced by the program by means of the
configuration file is an HDF file composed by 2 linked HDF tables. In Python,
you can open the tables via the `pandas` library:

```python
import pandas as pd
models_dataframe = pd.read_hdf(eval_result_file_path, key="models")
evaluations_dataframe = pd.read_hdf(eval_result_file_path, key="evaluations")
```

**Table "models"**

Each row denotes a model created with a specific combination of the values
input in the config file. Each column denote:

| Column        | Type | Description                                        |
|---------------|------|----------------------------------------------------|
| id            | int  | unique model id                                    |
| clf           | str  | the classifier name                                |
| param_`name1` | any  | the model parameters, prefixed with "param_"       |
| ...           | ...  |                                                    |
| param_`nameN` | any  |                                                    |
| training_set  | str  | The training-set path         |
| feat_`name1`  | bool | The training-set features used by the model, prefixed with "feat_" (true means: feature used) |
| ...           | ...  |                                                    |
| feat_`nameN`  | bool |                                                    |
| drop_na       | bool | If the classifier algorithm can not handle missing values (by default NaN and None) |
| inf_is_na     | bool | Whether +-Inf are considered NA                    |


**Table "evaluations"**

Each row denotes a model evaluation created with a specific combination of
the values input in this config file. Each column denote:

| Column               | Type | Description                                 |
|----------------------|------|---------------------------------------------|
| model_id             | int  | The unique model id (see table above)       |
| validation_set       | str  | The validation-set path                     |
| ground_truth_column  | str  | The validation set column denoting the true labels (passed as argument `y_true` in `sklearn.metrics` functions) |
| prediction_function  | str  | the function or classifier method used for prediction (passed as agument `y_pred`/`y_score` in `sklearn.metrics` functions) |
| evalmetric_`name1`   | any  | The evaluation metrics, prefixed with "evalmetric_") |
| ...                  | ...  |                                             |
| evalmetric_`nameN`   | any  |                                             |
