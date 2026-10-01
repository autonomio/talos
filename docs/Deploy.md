# Deploy

`talos.Deploy` selects a trained model from [Scan](Scan.md) or a RunResult and writes a local ZIP archive for [Restore](Restore.md). Import it from `talos`. Construction performs packaging immediately; it does not publish a model or start a service.

Prerequisites are completed results with a recoverable trained model, its installed [framework backend](Backends.md), and a writable destination. Only package and restore trusted models and caller code.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Deploy

deployment = Deploy(scan_object, 'experiment_name', metric='val_loss', asc=True)
```

The resulting `.path` names the archive. It can be transferred to another environment with compatible dependencies. Model selection sorts the named `metric` stably, dropping candidates whose selection metric is missing; ties retain result order.

NOTE: for a metric that is to be minimized, set `asc=True` or otherwise
you will end up with the model that has the highest loss.

## Arguments

| Parameter | Default | Description |
| --- | --- | --- |
| `scan_object` | required | Scan or RunResult with result rows and recoverable models. |
| `model_name` | required | Destination path; `.zip` is appended if absent. Parent directories are created. |
| `metric` | required | Existing result column used for model selection. |
| `asc` | `False` | Use `True` to minimize the selection metric. |
| `saved` | `False` | Recover from persisted artifacts rather than a retained live model. |
| `custom_objects` | `None` | Keras custom objects needed for model recovery. |
| `model_factory` | `None` | Torch factory, optionally paired with a constructor configuration dictionary. |

## Archive contents

The deploy package consists of:

- archive version, selected model and artifact metadata (`manifest.json`)
- backend-native trained model (`model.keras`, Torch state or joblib as appropriate)
- details and epoch histories (`details.json`, `history.json`)
- results of the experiment (`results.csv`)
- original parameters and samples of x/y data (`params.npy`, `x.npy`, `y.npy`)
- verified caller source snapshots when available

The package can be restored into a copy of the original Scan object using the `Restore()` command.

Only restore archives from trusted sources: legacy parameter/sample compatibility uses Python object serialization. This command packages a local archive; it does not publish a service.

## Results and failure boundaries

The signature is `Deploy(scan_object, model_name, metric, asc=False, saved=False, custom_objects=None, model_factory=None)`. The object exposes `.path`, `.model`, `.best_model` (the selected result index) and `.data`. Its compatibility `package()` and `save_model_as()` methods return the already-created archive path; they do not perform another deployment.

`x.npy` and `y.npy` contain up to the first 100 rows of each available training input or target, including nested arrays. Archives therefore contain samples of caller-owned data as well as executable model/source metadata; decide whether those samples may be transferred. Output paths are not exclusive: an existing ZIP at the requested destination can be overwritten.

Missing metric columns raise `KeyError`; no usable metric values or missing retained models raise `ValueError`. Source snapshot and model-artifact checksum failures stop packaging. Backend serialization errors and destination filesystem errors propagate. Python object serialization in parameters and samples makes trust a requirement; the ZIP format is a portability container rather than a sandbox.

## Read next

Use [Restore](Restore.md) on the destination, [Evaluate](Evaluate.md) before selecting a model, and [migration](Migration.md) for historical archive compatibility.
