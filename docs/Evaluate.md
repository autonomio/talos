# Evaluate

`talos.Evaluate` scores a trained model from a completed [Scan](Scan.md) or RunResult on caller-supplied held-out observations. Import it from `talos`. `Evaluate(scan_object)` stores the run; its `.evaluate()` method selects one fitted model and scores subsets without retraining.

Prerequisites are recoverable trained models, their installed [framework backend](Backends.md), and held-out features and targets with matching sample counts. Classification returns F1; continuous tasks return MAE. The `metric` argument selects the model from scan results; it does not change the held-out scoring formula.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Evaluate

# create the evaluate object
e = Evaluate(scan_object)

# perform the evaluation
scores = e.evaluate(x_test, y_test, task='multi_class', metric='val_loss',
                    asc=True, average='macro', folds=3, seed=17)
```

NOTE: It's very important to save part of your data for evaluation, and keep it completely separated from the data you use for the actual experiment. Choose an evaluation fraction suited to the dataset; the shared example reserves 20%. These folds score the same fitted model on held-out subsets; they do not retrain it.

## Interface

The method signature is `evaluate(x, y, task, metric, model_id=None, folds=5, shuffle=True, asc=False, saved=False, custom_objects=None, multi_input=False, print_out=False, average=None, model_factory=None, seed=None)`. It returns a list of `folds` floating-point scores. `Evaluate` holds `.scan_object` and its `.data` table but does not add score columns itself.

## Arguments

| Parameter | Default | Description |
| --- | --- | --- |
| `x`, `y` | required | Held-out features and truth labels, aligned by row. |
| `task` | required | Classification or continuous task, as described below. |
| `metric` | required | Scan result column used to select the fitted model. |
| `model_id` | `None` | Explicit result-row index; otherwise choose the best `metric` value. |
| `folds` | `5` | Number of held-out scoring subsets. |
| `shuffle` | `True` | Shuffle rows before splitting. |
| `asc` | `False` | Use `True` to minimize the selection metric. |
| `saved` | `False` | Load the selected model from persisted artifacts. |
| `custom_objects` | `None` | Keras objects required for model reconstruction. |
| `multi_input` | `False` | Set `True` for a list of feature arrays. |
| `print_out` | `False` | Print the mean and standard deviation of fold scores. |
| `average` | `None` | Task default F1 averaging; may be `binary`, `micro`, `macro`, `samples`, or `weighted` where supported by scikit-learn. |
| `model_factory` | `None` | Torch reconstruction factory when needed. |
| `seed` | `None` | Seed for shuffled fold assignment. |

The above arguments are for the <code>evaluate</code> attribute of the <code>Evaluate</code> object.

## Score semantics

| Task | Target/output handling | Score and units |
| --- | --- | --- |
| `binary` | Scalar output thresholded at `0.5`, or two-output predictions reduced with `argmax`. | F1 with binary averaging by default, from 0 to 1. |
| `multi_class` / `multiclass` | One-hot truth and predicted class distributions reduced with `argmax`. | Macro F1 by default, from 0 to 1. |
| `multi_label` | One-hot truth uses multiclass scoring; independent multi-hot truth uses per-label `0.5` thresholds. | Macro F1 by default, from 0 to 1. |
| `multilabel` / `multi_label_independent` | Independent per-label `0.5` thresholds. | Macro F1 by default, from 0 to 1. |
| `continuous` / `regression` | Raw predictions. | Mean absolute error, in the target's units. |

List or dictionary targets are scored output by output, then averaged for each subset. Every row enters exactly one subset, including a final uneven subset. Fold means are an unweighted mean of subset scores; for uneven folds this can differ from one score over all held-out rows.

`folds` must be an integer from 1 through the number of held-out rows. Invalid folds and unknown tasks raise `ValueError`; list feature inputs without `multi_input=True` raise `TypeError`. Missing metrics, unavailable saved models and shape errors propagate from model selection, restoration or scikit-learn. No leakage check can establish that the caller's held-out data was never used for training.

## Evaluate several candidates

`scan_object.evaluate_models(x_val, y_val, task, n_models=10, metric='val_acc', folds=5, shuffle=True, asc=False, saved=False, custom_objects=None, average=None, model_factory=None, multi_input=None, seed=None)` evaluates the top candidates selected by `metric`. It returns `None` and adds `eval_f1score_mean`/`eval_f1score_std`, or `eval_mae_mean`/`eval_mae_std`, to `.data`. Unselected rows receive missing values. With `multi_input=None`, a list of feature arrays is detected automatically.

These added columns are an in-memory table update; they do not rewrite the completed run's result files. Persist an exported table yourself when those post-run scores must be retained.

## Read next

Use [Predict](Predict.md) for inference with an explicit selection metric, [AutoPredict](AutoPredict.md) for candidate evaluation and winner prediction, or [Deploy](Deploy.md) for packaging.
