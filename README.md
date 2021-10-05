# skl-eval

Program to run full-scale evaluation in scikit-learn (hyperparameters 
optimization, features selection and metric scores computation)
from a single configuration file

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

    
This program is a simple wrapper on top of the usual 
[scikit-learn evaluation routines](https://scikit-learn.org/stable/modules/model_evaluation.html)
with the following feaures:

 1. Simple evaluation setup via a configuration file. Instead of implementing 
    your own code, input the classifier algorithm, parameters, features and 
    evaluation metrics, and let the program build all of their combinations 
    while showing progress bar and estimated time available on the terminal
 3. Portable and simple output showing evaluation results in tabular format
    (HDF file). The output structure is described at the bottom of the page
    and can be loaded it in your Python code via `pandas.read_hdf` to
    analyzed the evaluation results
    

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

<!-- [evaluation
  metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
 -->

### Configure the evaluation

If you don't have one already, create the example configuration file in YAML 
format (`config_file`). You can get the file [here (click on "Raw" to download it](https://github.com/rizac/skl-eval/blob/main/data/evalconfig.yaml) 
or copy it in a specific directory of your choice with the command:

```console
skl-eval-init <output_directory>
```
the `config_file` contains all documentation needed in order to be modified and configure 
your evaluation

#### Setup dataset paths

By default, in the `config_file` the paths of the datasets (training set and 
validation set) refer to three example datasets also copied in `<output_directory>` 
with `skl-eval-init`. The first thing to do then is to replace those paths with
the paths of your datasets.

The datasets preparation is a common and very important step in any machine
learning application and cannot be automatized, so it depends on the user needs.
Datasets must be saved as HDF files in table format and can be manipulated
from within Python with the `pandas` library, installed with the program.
Please consult the `config_file` for further details on how they should
be composed

#### Setup parameters and features

You can then configure the classifier type, its parameters,
the features to be used, and the evaluation metrics to compute. 
The program will then take care of building all combinations of 
parameters, features and metrics to calculate


#### (Optional) Configure custom prediction functions and evaluation metrics
 
By default, scikit-learn models implement their decision / prediction
functions via class methods (e.g. [predict, decision_function](https://scikit-learn.org/stable/glossary.html#methods)), and
the package offers all necessary [evaluation metrics](https://scikit-learn.org/stable/modules/model_evaluation.html). 
The prediction function and evaluation metric(s) are customizable 
by simply typing their Python path in the `config_file`
(e.g. "sklearn.metrics.log_loss")

You can also define your own prediction function and evaluation metrics in a
separate Python module, as long as the module is placed in the same directory 
of the `config_file`. The default `config_file` uses an external, fully-documented
Python module (also copied with `skl-eval-init`) that can be inspected


### Run the evaluation

Once setup, you can run the evaluation to a dedicated output file in HDF format
(You can always type `skl-eval --help` for details)

```console
skl-eval -c <config_file> <output_hdf_file>
```

## Evaluation output file description

(this section is available also at the bottom of the `config_file`)

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
| id            | int  | Unique model id                                    |
| clf           | str  | Python path of the scikit-learn classifier used    |
| param_`name1` | any  | classifier parameters, prefixed with "param_"      |
| param_`name2` | any  | (see above)                                        |
| ...           | ...  | ...                                                |
| training_set  | str  | The training-set path                              |
| feat_`name1`  | bool | The training-set features used by the model, prefixed with "feat_" (true means: feature used) |
| feat_`name2`  | bool | (see above)                                        |
| ...           | ...  | ...                                                |
| ground_truth_column | str  | The training set column denoting the true labels/numeric values used to fit the model (can be null if the classifier `fit` method needs no labels) |
| sample_weight_column | str  | The training set column denoting the optional sample weights to fit the model (empty: no weights) |
| drop_na       | bool | Remove values not avaliable (NA) from the training set. True if the classifier cannot handle NA values (by default NaN and None) |
| inf_is_na     | bool | Whether +-Inf should also be considered NA                    |


**Table "evaluations"**

Each row denotes a model evaluation created with a specific combination of
the values input in this config file. Each column denote:

| Column               | Type | Description                                 |
|----------------------|------|---------------------------------------------|
| model_id             | int  | The unique id of the model evaluated (see table above)       |
| validation_set       | str  | The validation set path. If this equals the model's training set path, then a cross validation was performed, splitting the training set by means of the object specified in the column `cv_splitter` |
| cv_splitter          | bool | In case of cross validation, it is the path of the scikit-learn splitter used. When null (check it with `pd.isna`), then the `validation_set` is provided as file and most likely distinct from the training set |
| ground_truth_column  | str  | The validation set column denoting the true labels (passed as argument `y_true` in `sklearn.metrics` functions) |
| sample_weight_column | str  | The validation set column denoting the optional sample weights (empty: no weights) |
| prediction_function  | str  | The function or classifier method used for prediction (passed as agument `y_pred`/`y_score` in `sklearn.metrics` functions) |
| metric_`name1`       | any  | The evaluation metrics, prefixed with "metric_")     |
| metric_`name2`       | any  | (see above)                                          |
| ...                  | ...  | ...                                                  |
