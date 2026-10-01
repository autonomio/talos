# Metrics

`talos.utils.metrics` provides lazy Keras-compatible training metrics. The same module is available as `talos.metrics.keras_metrics`. This page covers metrics supplied to `model.compile()`; [Evaluate](Evaluate.md) separately defines held-out F1 and MAE scoring after a sweep.

Install a [Keras or TensorFlow backend](Backends.md) before retrieving classification metric objects or calling tensor metric functions. The example uses the TensorFlow extra and the remotely acquired [Iris template](Templates.md). Available names are:

- matthews
- precision
- recall
- fbeta
- f1score
- mae
- mse
- rmae
- rmse
- mape
- msle
- rmsle

You can use these metrics as you would Keras metrics:

```python
import talos
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
x, y = talos.templates.datasets.iris()
model = Sequential([Input(shape=(4,)), Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', talos.utils.metrics.f1score])
history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
assert len(history.history['f1score']) == 1
```

If you would like to add new metrics to Talos, make a [feature request](https://github.com/autonomio/talos/issues/new) or create a [pull request](https://github.com/autonomio/talos/compare).

Classification metric objects accumulate counts across batches. `fbeta` is a stateless batch score; use `classification_metric(beta=...)` for an epoch-wide score. Continuous root helpers reduce over each example’s target axis before Keras averages samples.

## Classification interfaces

`matthews`, `precision`, `recall` and `f1score` are fresh Keras Metric objects when accessed through the module. They accumulate true-positive, false-positive and false-negative counts over batches and support sample weights. Use a separate metric instance for each independently compiled model.

`classification_metric(kind='f1score', beta=1, num_classes=None, name=None)` constructs an explicit object. Supported kinds are `f1score`, `precision`, `recall` and `matthews`; `name` defaults to `kind`, and the output width is inferred on first update unless `num_classes` is provided. Negative `beta` raises `ValueError`.

Scalar binary predictions and truth are thresholded at `0.5`. Multiclass predictions use `argmax`; truth may be integer labels or one-hot arrays. Precision, recall and F-beta average class scores equally; Matthews uses the binary or multiclass correlation formula. Values are dimensionless: F1, precision and recall range from 0 to 1, and Matthews ranges from -1 to 1 when defined. Zero denominators use a small numeric floor.

`fbeta(y_true, y_pred, beta=1)` is a stateless batch score rather than an epoch accumulator. It uses the same classification conversion, requires a nonnegative beta and returns a scalar. For epoch-wide F-beta, use `classification_metric(beta=...)` instead. These multiclass helpers do not implement independent multilabel thresholds for outputs wider than one; use a task-appropriate framework metric for that case.

## Continuous interfaces and units

Continuous functions accept `y_true` and `y_pred`, align a trailing singleton dimension where necessary, and return one value per example after reducing the final target axis. Keras then averages those values over samples.

| Function | Calculation | Unit |
| --- | --- | --- |
| `mae` | Mean absolute target error. | Target unit. |
| `mse` | Mean squared target error. | Target unit squared. |
| `rmae` | Square root of per-example MAE. | Square root of target unit. |
| `rmse` | Square root of per-example MSE. | Target unit. |
| `mape` | Mean absolute relative error times 100; true magnitude floored at `1e-7`. | Percent. |
| `msle` | Mean squared difference of `log(1 + value)` after values are floored at `1e-7`. | Dimensionless. |
| `rmsle` | Square root of per-example MSLE. | Dimensionless. |

`rmae` means root MAE, not relative MAE. Averaging per-example RMSE is not the same operation as taking the square root of a dataset-wide MSE. Choose the metric matching the quantity your experiment intends to estimate.

## Results and restoration

The example returns one `f1score` history value after one epoch; Scan logs final epoch values and retains their complete histories. Native Torch loops can return their own metric histories without using Keras metrics.

Tensor shape, dtype and unsupported framework-operation errors propagate. Classification needs an inferable output width. When restoring a model that uses custom metric classes, supply the objects required by the archive; [Restore](Restore.md) loads models without compilation for prediction.

## Read next

See [Evaluate](Evaluate.md) for held-out scoring, [Analyze](Analyze.md) for result-column analysis, and [Backends](Backends.md) for callback result normalization.
