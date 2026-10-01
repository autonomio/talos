# Predict()

In order to identify the best model from a given experiment, or to perform predictions with model/s, the [Predict()](https://github.com/autonomio/talos/blob/master/talos/commands/predict.py) command can be used.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
predictor = talos.Predict(scan_object)
probabilities = predictor.predict(x_test, metric='val_loss', asc=True)
```

### Predict Properties

**`predict`** makes probability predictions on `x` which has to be in the same form as the input data used in the `Scan()` experiment.

```python
predictor.predict(x_test, metric='val_loss', asc=True)
```

<hr>

**`predict_classes`** makes class predictions on `x` which has to be in the same form as the input data used in the `Scan()` experiment.

```python
predictor.predict_classes(x_test, metric='val_loss', asc=True, task='multi_class')
```

### Predict Arguments

### Predict.predict Arguments

Parameter | Default | Description
--------- | ------- | -----------
`x` | NA | the predictor data x
`model_id` | None | the model_id to be used
`metric` | required | the metric against which the validation is performed
`asc` | required | should be True if metric is a loss
`task`| required for predict_classes | 'binary', 'multi_class', 'multilabel', or 'continuous'
`saved` | bool | if a model saved on local machine should be used
`custom_objects` | dict | if the model has a custom object, pass it here
`model_factory` | callable or None | Reconstructs a saved Torch model when its architecture cannot be recovered from the archive.
