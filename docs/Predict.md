# Predict()

In order to identify the best model from a given experiment, or to perform predictions with model/s, the [Predict()]([Reporting()](https://github.com/autonomio/talos/blob/master/talos/utils/predict.py)) command can be used.

```python
p = Predict('scan_object')

p.predict(x)
```

### Predict Properties

**`predict`** makes probability predictions on `x` which has to be in the same form as the input data used in the `Scan()` experiment.

```python
scan_object.data
```

<hr>

**`predict_classes`** makes class predictions on `x` which has to be in the same form as the input data used in the `Scan()` experiment.

```python
scan_object.data
```

### Predict Arguments

### Predict.predict Arguments

Parameter | Default | Description
--------- | ------- | -----------
`x` | NA | the predictor data x
`model_id` | None | the model_id to be used
`metric` | None | the metric against which the validation is performed
`asc` | None | should be True if metric is a loss
`task`| NA | One of the following strings: 'binary' or 'multi_class'
`saved` | bool | if a model saved on local machine should be used
`custom_objects` | dict | if the model has a custom object, pass it here
