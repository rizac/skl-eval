# skl-eval
Program to run iteratively scikit learn evaluations and save them in tabular HDf file

## Installation
Git clone the repository, create virtual environment, move to the cloned
directory and then as usual 
```console
pip install -e .
```
(-e is optional  and makes this package editable, meaning that any new 
`git pull` automatically updates the package without re-installation needed)

## Test:
```console
python -m unittest -f
```

## Usage

### Prepare your datasets

The datasets can be created as `DataFrame` objects with the `pandas` library
(installed with the  program) and then saved as HDF via the `DataFrame.to_hdf`
method

To run the program you need:

- One or more trainingset, with features arranged by column(s), and 
  instances arranged by rows
- One or more testsets, with the same features as the training set(s), an
  additional "groun truth" column with the real values (e.g. class labels,
  numeric scores) and, optionally, a column named 'sample_weight': this column
  if present will be used in the sample_weight argument in most [evaluatio
  metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
  

### Configure the evaluation(s)
Open the file (TBD), change the configuration therein

### Run the program
```console
skl-eval -c <config_file> <output_hdf_file>
```
