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

Parameter | Description
--------- | -----------
`scan_object` | The resulting class object from `Scan()`
