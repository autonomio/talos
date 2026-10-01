# AutoPredict

`AutoPredict()` automatically handles the process of finding the best models from a completed `Scan()` experiment, evaluates those models, and uses the winning model to make predictions on input data.

```python
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
x, y = load_iris(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x.astype('float32'), y, test_size=.2, stratify=y, random_state=17)
p = talos.autom8.AutoParams(task='multi_class', network=False, resample_params=1).params
p.update({'epochs': [1], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['sparse_categorical_crossentropy'],
          'kernel_initializer': ['glorot_uniform'], 'activation': ['relu'],
          'shapes': ['brick'], 'lr': [1.]})
p['activation'] = ['relu', 'elu']
model = talos.autom8.AutoModel(task='multi_class', experiment_name='iris_autopredict').model
scan_object = talos.Scan(x_train, y_train, params=p, model=model,
                         experiment_name='iris_autopredict', round_limit=2, seed=17)
scan_object = talos.autom8.AutoPredict(scan_object, x_val=x_test, y_val=y_test,
                                     x_pred=x_test, task='multi_class', metric='val_f1score',
                                     n_models=2, folds=2, asc=False)
assert scan_object.preds_classes.shape == (len(x_test),)
```

NOTE: the input data must be in same format as 'x' that was used in `Scan()`.
Also, `x_val` and `y_val` should not have been exposed to the model during the
`Scan()` experiment.

`AutoPredict()` will add four new properties to `Scan()`:

**`preds_model`** contains the winning trained model
**`preds_parameters`** contains the hyperparameters for the selected model
**`preds_probabilities`** contains the prediction probabilities for `x_pred`
**`preds_classes`** contains the predicted classes for `x_pred`.

## AutoPredict Arguments

Argument | Input | Description
--------- | ------- | -----------
`scan_object` | class object | the class object returned from `Scan()`
`x_val` | array or list of arrays | validation data features
`y_val` | array or list of arrays | validation data labels
`x_pred` | array or list of arrays | prediction data features
`task` | string | 'binary', 'multi_class', 'multi_label', or 'continuous'
`metric` | None | the metric against which the validation is performed
`n_models` | int | number of promising models to be included in the evaluation process
`folds` | None | number of held-out scoring folds; models are not retrained
`shuffle` | None | if data is shuffled before splitting
`asc` | None | should be True if metric is a loss
