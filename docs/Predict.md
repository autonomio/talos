# Predict

`talos.Predict` selects a trained model from [Scan](Scan.md) or a RunResult and performs inference on caller-supplied features. Import it from `talos`. Construction stores the run; prediction happens when `.predict()` or `.predict_classes()` is called.

Prerequisites are a completed run with a recoverable model and its installed [framework backend](Backends.md). Apply the same preprocessing and feature structure used for training; this wrapper does not reconstruct caller-owned preprocessing automatically.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
predictor = talos.Predict(scan_object)
probabilities = predictor.predict(x_test, metric='val_loss', asc=True)
```

## Prediction methods

**`predict`** returns the trained model's raw outputs for `x`. These are probabilities only when the model produces probabilities; regression predictions and logits remain unchanged.

```python
predictor.predict(x_test, metric='val_loss', asc=True)
```

<hr>

**`predict_classes`** makes class predictions on `x` which has to be in the same form as the input data used in the `Scan()` experiment.

```python
predictor.predict_classes(x_test, metric='val_loss', asc=True, task='multi_class')
```

## Arguments

| Parameter | Default | Description |
| --- | --- | --- |
| `x` | required | Features in the model's expected input structure. |
| `metric` | required | Scan result column used to select a model. |
| `asc` | required | `True` for lower metric values, `False` for higher values. |
| `model_id` | `None` | Explicit result-row index; otherwise select by metric. |
| `task` | required for `predict_classes` | Class conversion rule described below. |
| `saved` | `False` | Recover from persisted artifacts rather than a retained live model. |
| `custom_objects` | `None` | Keras objects needed for reconstruction. |
| `model_factory` | `None` | Torch reconstruction factory when needed. |
| `**kwargs` | empty | `.predict()` only: forwarded to backend prediction, such as Keras `batch_size`. |

## Signatures and class conversion

The constructor is `Predict(scan_object)`. Its methods are `predict(x, metric, asc, model_id=None, saved=False, custom_objects=None, model_factory=None, **kwargs)` and `predict_classes(x, metric, asc, task, model_id=None, saved=False, custom_objects=None, model_factory=None)`.

| Task | Class conversion |
| --- | --- |
| `binary` | `argmax` for two outputs; otherwise threshold each value at `0.5`. |
| `multi_class`, `multiclass`, `multi_label` | `argmax` across the last axis for multidimensional output; one-dimensional output becomes integers. |
| `multilabel`, `multi_label_independent` | Independent `0.5` thresholds. |
| `continuous`, `regression` | Return the raw predicted values. |

The historical `multi_label` spelling means exclusive class conversion here. Use `multilabel` or `multi_label_independent` for independent labels. Thresholds require suitably scaled outputs; Torch logits are not transformed into probabilities automatically. Structured multi-output predictions can be obtained through `.predict()`; class conversion expects a single array.

Prediction returns outputs and does not add columns to the result table or retrain the model. Tied selection metrics preserve trial order, and missing values are excluded. Missing metric columns raise `KeyError`; an unusable selection or unretained model raises `ValueError`. Unknown class tasks raise `ValueError`, and framework shape or restoration errors propagate.

## Read next

See [Evaluate](Evaluate.md) for held-out scoring, [Backends](Backends.md) for native inference behavior, or [Deploy](Deploy.md) to archive the selected trained model.
