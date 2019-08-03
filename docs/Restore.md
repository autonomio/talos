# Restore()

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`.
å
```python
from talos import Restore

restore = Restore('experiment_name.zip')
```
NOTE: In the `Deploy()` phase, '.zip' is automatically added to the deploy package file name and must be added here manually.

## Restore Arguments

Parameter | Default | Description
--------- | ------- | -----------
`path_to_zip` | None | full path to the `Deploy` asset zip file


## Restore Properties

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`. The object consists of:

- details of the scan
- model
- results of the experiment
- sample of x data
- sample of y data

**`details`** returns a pandas DataFrame with various meta-information about the experiment.

```python
restore.details
```
<hr>

**`model`** returns a Keras model ready to rock.

```python
restore.model
```
<hr>

**`params`** returns the params dictionary used in the experiment.

```python
restore.params
```
<hr>

**`results`** returns a pandas DataFrame with the results for the experiment together with the hyperparameter permutation details.

```python
restore.results
```

<hr>

**`x`** returns a small sample of the data (features) used for training the model.

```python
restore.x
```

<hr>

**`y`** returns a small sample of the data (labels) used for training the model.

```python
restore.y
```

<hr>


`details` | The class object returned by Scan() upon completion of the experiment.
`model` | Input data (features) in the same format as used in Scan(), but should not be the same data (or it will not be much of validation).
`params` | Input data (labels) in the same format as used in Scan(), but should not be the same data (or it will not be much of validation).
`results` | The number of models to be evaluated. If set to 10, then 10 models with the highest metric value are evaluated. See below.
`x` |
`y` | A
