# AutoPredict

`AutoPredict()` automatically handles the process of finding the best models from a completed `Scan()` experiment, evaluates those models, and uses the winning model to make predictions on input data.

```python
scan_object = talos.autom8.AutoPredict(scan_object, x_val=x, y_val=y, x_pred=x)
```

NOTE: the input data must be in same format as 'x' that was used in `Scan()`.
Also, `x_val` and `y_val` should not have been exposed to the model during the
`Scan()` experiment.

`AutoPredict()` will add four new properties to `Scan()`:

**`preds_model`** contains the winning Keras model (function)
**`preds_parameters`** contains the hyperparameters for the selected model
**`preds_probabilities`** contains the prediction probabilities for `x_pred`
**`predict_classes`** contains the predicted classes for `x_pred`.

## AutoPredict Arguments

Argument | Input | Description
--------- | ------- | -----------
`scan_object` | class object | the class object returned from `Scan()`
`x_val` | array or list of arrays | validation data features
`y_val` | array or list of arrays | validation data labels
`y_pred` | array or list of arrays | prediction data features
`task` | string | 'binary', 'multi_class', 'multi_label', or 'continuous'
`metric` | None | the metric against which the validation is performed
`n_models` | int | number of promising models to be included in the evaluation process
`folds` | None | number of folds to be used for cross-validation
`shuffle` | None | if data is shuffled before splitting
`asc` | None | should be True if metric is a loss
