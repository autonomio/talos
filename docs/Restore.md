# Restore()

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Restore

deployment = talos.Deploy(scan_object, 'experiment_name', metric='val_loss', asc=True)
restore = Restore(deployment.path)
```
NOTE: In the `Deploy()` phase, '.zip' is automatically added to the deploy package file name and must be added here manually.

## Restore Arguments

Parameter | Default | Description
--------- | ------- | -----------
`path_to_zip` | required | full path to the `Deploy` asset zip file
`custom_objects` | dict or None | Keras custom layers, losses or metrics needed by the model.
`model_factory` | callable or None | Reconstructs a saved Torch model when its architecture cannot be recovered from the archive.


## Restore Properties

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`. The object consists of:

- details of the scan
- model
- results of the experiment
- sample of x data
- sample of y data

**`details`** returns a pandas Series with various meta-information (historical archives may return a DataFrame) about the experiment.

```python
restore.details
```
<hr>

**`model`** returns the restored trained backend model, ready for prediction without retraining.

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


Property | Description
-------- | -----------
`details` | Experiment metadata.
`model` | The selected trained model.
`params` | Original parameter candidates.
`results` | Trial results and parameter values.
`x` | Sample of training features.
`y` | Sample of training labels.
