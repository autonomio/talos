# Restore

`talos.Restore` reads a trusted [Deploy](Deploy.md) ZIP archive and restores the selected fitted model, result table and experiment assets. Import it from `talos`. Construction extracts the archive and loads the model immediately; it does not run training.

Prerequisites are a readable archive, the matching installed [framework backend](Backends.md), and any custom Keras objects or Torch reconstruction factory the model needs. Restoring an archive can load executable caller code and serialized Python objects; use trusted archives only.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Restore

deployment = talos.Deploy(scan_object, 'experiment_name', metric='val_loss', asc=True)
restore = Restore(deployment.path)
```

Deploy adds `.zip` to the destination when needed. Restore accepts the actual ZIP path, such as `deployment.path` in this example; it does not append an extension.

## Arguments

| Parameter | Default | Description |
| --- | --- | --- |
| `path_to_zip` | required | Path to the Deploy ZIP archive. |
| `custom_objects` | `None` | Keras custom layers, losses or metrics needed by the model. |
| `model_factory` | `None` | Torch constructor factory, optionally paired with a configuration dictionary. |

## Restored properties

The `Deploy()` .zip package can be read back into a copy of the original experiment assets with `Restore()`. The object consists of:

- details of the scan
- model
- results of the experiment
- sample of x data
- sample of y data

**`details`** returns a pandas Series with various meta-information (historical archives may return a DataFrame) about the experiment.

```python
restore.details
```

<hr>

**`model`** returns the restored trained backend model, ready for prediction without retraining.

```python
restore.model
```

<hr>

**`params`** returns the params dictionary used in the experiment.

```python
restore.params
```

<hr>

**`results`** returns a pandas DataFrame with the results for the experiment together with the hyperparameter permutation details.

```python
restore.results
```

<hr>

**`x`** returns a small sample of the data (features) used for training the model.

```python
restore.x
```

<hr>

**`y`** returns a small sample of the data (labels) used for training the model.

```python
restore.y
```

<hr>

| Property | Description |
| -------- | ----------- |
| `details` | Experiment metadata. |
| `model` | The selected trained model. |
| `params` | Original parameter candidates. |
| `results` | Trial results and parameter values. |
| `x` | Sample of training features. |
| `y` | Sample of training labels. |

## Archive compatibility and lifetime

The signature is `Restore(path_to_zip, custom_objects=None, model_factory=None)`. `.data` aliases `.results`, so the restored result table can be inspected with [Analyze](Analyze.md). `.round_history` contains recorded epoch histories for Talos 2 archives; historical JSON/H5 archives use an empty list. A restored archive contains one selected trained model, not every model from the original sweep.

Talos 2 archives declare `talos_archive_version=2` in `manifest.json`. They hydrate verified source snapshots before loading the backend-native artifact. Historical archives without that manifest are accepted through the original single model JSON and H5 weights layout. Keras models load with `compile=False`; prediction works, but further fitting requires the caller to compile the model again.

Extraction uses a temporary directory available as `.run_dir` while the Restore object retains its temporary-directory owner. Keep the object alive while using files or source snapshots from that directory. `.x` and `.y` are samples rather than the complete training dataset and retain their recorded nested structure in version 2 archives.

ZIP members escaping the extraction directory, unsupported archive versions, missing or changed source snapshots and artifact checksum mismatches fail restoration. Historical archives must contain exactly one model JSON. Missing custom objects or a required Torch factory, incompatible framework versions and malformed archive members raise their native loading errors. Successful restoration verifies recorded assets; it does not establish the scientific validity of the original experiment.

## Read next

Use [Predict](Predict.md) for inference, [Analyze](Analyze.md) for result inspection, and [migration](Migration.md) for archive and resumable-run boundaries.
