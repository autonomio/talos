## Predict()

In order to identify the best model from a given experiment, or to perform predictions with model/s, the [Predict()]([Reporting()](https://github.com/autonomio/talos/blob/master/talos/utils/predict.py)) command can be used.

> Using predict

```python
p = Predict('scan_object')

# returns model_id for best performing model
r.best_model(metric='val_fmeasure')

# returns predictions for input x
r.predict(x)

# performs a 10-fold cross-validation for multi-class prediction
r.evaluate(x, y, folds=10, average='macro')
```

### Predict Functions

See docstring for each function for a more detailed information, and the required input arguments.

**`load_model`** Loads the Keras model with weights so it can be used in the local environment for predictions or other purpose. Requires `model_id` as argument. The `model_id` corresponds with the round in the experiment.

**`best_model`** Identifies the `model_id` for the best performing model based on a given metric (e.g. 'val_fmeasure').

**`predict`** Makes predictions based on input `x` and `model_id`. If `model_id` is not given, best model will be used.

**`predict_classes`** Same as predict, but predicts classes.

**`evaluate`** Evaluates models using a k-fold crossvalidation.

### Predict Arguments

Parameter | Default | Description
--------- | ------- | -----------
`scan_object` | None | the object from Scan() after experiment is completed
