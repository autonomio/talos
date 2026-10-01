# AutoPredict

`talos.autom8.AutoPredict` selects candidates from a completed [Scan](Scan.md), scores them on held-out observations, and predicts with the winning trained model. Import it through `talos.autom8`. It is a function returning the same Scan object after adding evaluation and prediction fields.

Prerequisites are a completed run with retained models, its [framework backend](Backends.md), and evaluation data kept outside the scan. The standalone example below additionally requires scikit-learn, included with the Talos core, and TensorFlow for the default AutoModel callback.

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
**`preds_parameters`** contains the selected result row as a pandas Series, including metrics and hyperparameter columns
**`preds_probabilities`** contains raw prediction outputs for `x_pred`
**`preds_classes`** contains the predicted classes for `x_pred`.

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `scan_object` | required | Completed Scan or RunResult with recoverable trained models. |
| `x_val`, `y_val` | required | Held-out features and labels for candidate evaluation. |
| `x_pred` | required | Features on which to predict with the winner. |
| `task` | required | `binary`, `multi_class`, `multi_label`, or `continuous`; see task semantics in [Evaluate](Evaluate.md). |
| `metric` | `'val_acc'` | Existing result column for the initial candidate ranking. |
| `n_models` | `10` | Maximum candidate models to evaluate. |
| `folds` | `5` | Held-out scoring subsets; models are not retrained. |
| `shuffle` | `True` | Shuffle held-out rows before splitting. |
| `asc` | `False` | Initial ranking direction; use `True` to minimize `metric`. |
| `custom_objects` | `None` | Keras objects required to reconstruct selected models. |
| `model_factory` | `None` | Torch reconstruction factory for model recovery during evaluation. |
| `average` | `None` | F1 averaging choice, with task-specific defaults. |

## Selection and side effects

The function adds `eval_f1score_mean` and `eval_f1score_std` for classification, or `eval_mae_mean` and `eval_mae_std` for continuous tasks. It then maximizes held-out F1 or minimizes held-out MAE, independently of the initial Scan ranking metric. Models are not fitted again. `preds_probabilities` contains raw model outputs; for regression these are predicted values rather than probabilities.

The example returns two evaluated rows and one predicted class for each held-out Iris sample. The supplied `x_val`/`y_val` select a model, so use another untouched test set when reporting a final generalization result. The `multi_label` task uses the historical exclusive-class conversion for `.preds_classes`; use `multilabel` or `multi_label_independent` for independent-label predictions, as described in [Predict](Predict.md). AutoPredict's shuffle has no seed argument; use [Evaluate](Evaluate.md) directly when deterministic fold assignments are required.

Missing metric columns, unusable candidates, invalid fold counts and model restoration failures propagate from Evaluate or model selection. `model_factory` is forwarded to evaluation; for a saved Torch winner that lacks a recorded factory, set `scan_object.model_factory` as well so final winner recovery uses the same factory.

## Read next

See [Evaluate](Evaluate.md) for score semantics, [Predict](Predict.md) for explicit inference, and [Deploy](Deploy.md) for a transferable archive.
