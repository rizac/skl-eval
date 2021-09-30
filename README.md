# skl-eval

Program to run iteratively scikit-learn evaluations from pre-built
training and validation set(s): implement your own features, parameters 
and evaluation metrics in a configuration file, and let the program run 
iteratively any given combination of them. 

```console
$ skl-eval -c ./path/to/my-config.yaml /path/to/my-evaluation-result.hdf
   Reading configuration file
   ==========================
             training set(s):       1  ×
   parameters combination(s):      27  ×
     features combination(s):    4095  ×
           validation set(s):       4  =
 ----------------------------
 Total number of evaluations:  442260

Computing  [oooooooooooooooooooooooooooo........]   80%  04:43:12
```
<sup>excerpt of the terminal during evaluation</sup>
<hr>

This program assumes that you are already familiar with the
[scikit-learn evaluation routines](https://scikit-learn.org/stable/modules/model_evaluation.html),
and offers an alternative solution in those cases when:

 1. A standard cross validation is not sufficient and thus
    separate training and validation set(s) are pre-built and available
 2. You want to avoid the burden of implementing your own code, focusing on the
    configuration of your parameters, letting the program build all features 
    and parameters combinations with a single command while showing 
    progress bar and estimated time available on the terminal
 3. You want to save your evaluations in a portable and simple tabular format
    (HDF file) whose structure is described at the bottom of the page
    and can be loaded in your Python code for analysis via `pandas.read_hdf`
    

## Installation

Create a directory where to clone the project, `cd` in it and then
as usual:

```console
git clone <repository_url> .
python3 -m venv <env_dir>      # <- create virtual Python environment
source <env_dir>/bin/activate  # <- activate it (if not already activated)
pip install -e .               # <- install the program
```
(-e is optional  and makes this package editable, meaning that any new 
`git pull` automatically updates the package without re-installation needed)

Notes: 
 - **The virtual environment must be activated any time you want to use 
   the program** (type `deactivate` on the terminal to de activate it)

 - For developers who want to run tests:

   ```console
   python -m unittest -f
   ```

## Usage

### Prepare your datasets

The datasets can be created as `DataFrame` objects with the `pandas` library
(installed with the  program) and then saved as HDF via the `DataFrame.to_hdf`
method. To run the program you need:

- One or more training set, with features arranged by column(s), and 
  instances arranged by rows
- One or more validation sets, with the same features as the training set(s), 
  and an additional "ground truth" column with the real values, e.g. class 
  labels,  numeric scores. The ground truth column can have any name. 
  Note that if a column named 'sample_weight' is present, it will be used in 
  the `sample_weight` argument of most [evaluation
  metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
  

### Configure the evaluation

If you don't have one already, create the example configuration file in YAML 
format:

```console
skl-eval-init <output_directory>
```
creates the file `<output_directory>/evalconfig.yaml` which contains all
documentation needed and can thus be renamed and/or edited to configure your 
evaluation:

#### Setup dataset paths

By default, in the created config the training set and test set paths
refer to three example datasets also provided in `<output_directory>`.
The first thing to do then is to replace those paths with the paths of your 
datasets


#### Setup parameters and features

You can configure the classifier type, its parameters,
the features to be used, and the evaluation metrics to compute. 
The program will then take care of building all combinations of 
parameters, features and metrics to calculate


#### (Optional) Configure custom prediction functions and evaluation metrics
 
By default, scikit-learn models implement their decision / prediction
functions via class methods (e.g. `predict`, `decision_function`), and
the package offers all necessary [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html). 
The prediction function and evaluation metric(s) are customizable 
by simply typing their Python path in the configuration file
(e.g. "sklearn.metrics.log_loss")

You can also define your own prediction function and evaluation metrics in a
separate Python module, as long as the module is placed in the same directory 
of the configuration file, and then simply type their paths
in the evaluation config.

The example configuration file by default employs user-defined functions
implemented in a separate Python module. Both files provide all the necessary
documentation in case of help


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
| sample_weight_column | str  | The training set column denoting the optional sample weights to fit the model (empty: no weights) |


**Table "evaluations"**

Each row denotes a model evaluation created with a specific combination of
the values input in this config file. Each column denote:

| Column               | Type | Description                                 |
|----------------------|------|---------------------------------------------|
| model_id             | int  | The unique model id (see table above)       |
| validation           | str  | The validation-set path, or, in case of cross validation, the scikit-learn splitter class |
| ground_truth_column  | str  | The validation set column denoting the true labels (passed as argument `y_true` in `sklearn.metrics` functions) |
| prediction_function  | str  | The function or classifier method used for prediction (passed as agument `y_pred`/`y_score` in `sklearn.metrics` functions) |
| sample_weight_column | str  | The validation set column denoting the optional sample weights (empty: no weights) |
| metric_`name1`       | any  | The evaluation metrics, prefixed with "evalmetric_") |
| ...                  | ...  |                                             |
| metric_`nameN`       | any  |                                             |
