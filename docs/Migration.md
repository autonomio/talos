# Talos 2 migration

Talos 2 uses one search queue, executor and artifact format for `Scan`, native SFDs and the CLI. The core manages parameter experiments; data acquisition, model architecture and training remain caller-owned.

## Prerequisites and scope

Use an existing Talos callback in a supported framework environment; see
[Installation](Install_Options.md) for extras and dependency floors. Keep a copy
of existing run artifacts before changing environments. This guide covers the
established Python interface, an incremental SFD port and recovery boundaries.
It does not convert a model architecture or its training loop.

## Existing applications

Keep `model(x_train, y_train, x_val, y_val, params)` returning `(history, trained_model)` and the current `talos.Scan(x, y, params, model, experiment_name, ...)` call. Positional order and established keywords remain supported. Framework-specific history objects or dictionaries are accepted. The documented Torch `(network_with_history, network.parameters())` return also works.

When a parameter name collides with a metric or timing column (for example `loss`), numeric metrics keep their names and parameters use `param__loss`. `.parameter_columns` records the mapping; callbacks, reducers and model recovery still receive original parameter names. The mapping and exact per-trial parameters survive schema changes and resume.

Results retain pandas `.data`, `.round_history`, `.round_times`, `.learning_entropy`, `.params`, `.details`, `.saved_models` and `.saved_weights`. `best_model()` returns the trained model; `Analyze.best_params()` returns an array. `Restore.results` remains available, with `.data` as an additional alias. `Reporting` aliases `Analyze`.

`save_models=True` persists native framework artifacts. With the historical `save_weights=True` default, memory-mode scans also retain durable trained artifacts so completed trials survive process exit. `save_models=False, save_weights=False` explicitly keeps metrics without trained weights. Saved artifacts, metadata and CSVs live in a unique run directory under `experiment_name`, or an explicit `experiment_dir`.

Python objects are passed directly to training. Lists/dictionaries of aligned inputs and outputs are split using the same sample indices. Streams, generators, images, sequences and tensors remain caller-owned. Use a user `prep` function for arbitrary structures or explicit splits.

Reducers, limits, shards and samplers remain supported. Legacy grid order and deliberate duplicate trials are preserved; each duplicate has a separate trial ID. Mutable `ParamSpace` controls the same queue as native feedback. Live `talos_strategy.py`, Gamify and native `interventions.json` controls apply between trials and create audit records. Live model, train/validation data, reducer and artifact-policy changes are checkpointed; data changes record fingerprints in the audit. Only changed live data is persisted. Dense arrays/tensors preserve types and dtypes on resume; changed opaque streams need a reconstructible caller path. Distributed workers use distinct, stable trial namespaces, including intentional duplicates.

## SFD port

Wrap an existing callback without changing it:

```python
# my_sfd.py
from my_models import existing_model


def params():
    return {'epochs': [5, 10]}


def prep(data, round_params):
    return data


def model(data, round_params):
    return existing_model(data['x_train'], data['y_train'],
                          data['x_val'], data['y_val'], round_params)
```

Call `talos.run('my_sfd', data=my_splits)` from Python. For CLI runs, implement your own loading/preparation inside `prep`, or provide manifest context understood by your code. The executor never selects a reader or acquires observations. Educational `talos.templates.datasets` functions remain explicit opt-in helpers.

## Scientific records and resume

Runs store `metadata.json`, `round_data.jsonl`, `results.csv`, `checkpoint.json`, `models/`, source snapshots and `audit.jsonl` when interventions occur. Records link manifest, experiment, parameter-hash and trial identities to histories, prepared split fingerprints, seeds and trained artifact hashes. Metadata records Python/platform/dependency versions, source hashes and available Git revisions. A declared backend is loaded and seeded before preparation/training; SFD templates declare it. Caller code controls backend deterministic operations and device-specific nondeterminism.

Resume checks configuration, code, environment and supplied data, and replays preparation for the first completed split to detect changed prepared data. Completed trials are loaded from artifacts. Verification failures leave the committed checkpoint intact; partially staged trials roll back queue, aliases, histories and records together. A signal or interrupted current trial rolls its queue state back. Checkpoints define committed completed trials; a crash discards an uncommitted log tail. Use the default interval of one for the smallest recovery window. Within-epoch training recovery belongs to the caller framework.

Importable callable parameter values are recorded as module/qualified-name references. Local lambdas can run in the current process but explicitly fail portable resume when their behavior cannot be reconstructed. For opaque or streaming data, pass `data_fingerprint` with a stable caller-supplied identity; the core does not consume a stream to hash it. Local helper modules and callable candidates are snapshotted with their package layout and checksums. Retain installed dependencies and any dynamically acquired code outside that recorded graph.

Torch persistence uses `state_dict` plus an importable factory and plain constructor configuration. Return `{'model': network, 'history': history, 'factory': (factory, config)}` or set `network.talos_factory` / `network.talos_config`; alternatively supply `model_factory` explicitly. Nested classes remain usable in memory; fresh-process restoration needs a factory. Keras uses native `.keras` files and supports `custom_objects` on selection and restoration.

New Deploy archives are versioned native packages. Historical Keras JSON/H5 archives remain readable, including the compatibility bridge for modern Keras. Loading archives and model factories requires trusting the source code and serialized model.

## Port procedure and expected result

1. Run the existing callback through `Scan` with its original data and parameters.
2. Preserve that callback and import it from the SFD wrapper above.
3. Pass the same explicit splits to `talos.run` and declare the same objective
   and direction. Begin with a bounded parameter set.
4. Inspect the result's `.data`, select a trained artifact and compare predictions
   on the same held-out observations. A successful port produces completed-trial
   records and a restorable artifact; scores need not be bit-identical across
   framework versions or devices.
5. Exercise [Deploy](Deploy.md) and [Restore](Restore.md) in a fresh process before
   retiring the old environment. Check source dependencies and custom objects.

A source, configuration or data verification failure must be resolved before
resume. Do not edit a checkpoint to force acceptance. An opaque stream requires
a reconstructible caller path and a stable identity; a missing importable Torch
factory or Keras custom object prevents portable restoration.

## Corrected behavior

Talos 2 corrects confirmed defects: stateful F1 is invariant to batch partitioning; binary outputs shaped `(n, 1)` are thresholded correctly; multiclass probabilities use class selection; objective direction is respected; evaluation folds include remainder rows and reject empty folds; shuffled features and labels stay aligned; reducer windows/cadence, live-file reload and Gamify annotations behave consistently; later-added metrics are retained in CSVs. Fold evaluation scores an already-trained model on held-out partitions; it does not retrain models as cross-validation would.

Core imports no DL or plotting backend. Python 3.10–3.13, modern optional frameworks and a separate TensorFlow 2.14 / NumPy 1.26 lane replace the obsolete Python 2/3.5 metadata. See [installation](Install_Options.md) and [SFD/CLI usage](SFD_and_CLI.md).

## Legacy security backports

The [owned legacy wheels](Developer/Security-Backports.md) retain TensorFlow 2.14.1,
the five-argument callback and legacy optimizers. Safe loading now rejects Lambda
bytecode by default, external vocabulary paths, HDF5 links/external or virtual
storage, oversized dataset allocation, archive traversal and archive links.
NPZ object arrays require an explicit trusted loading scope. Ordinary model
weights, vocabulary assets and registered/custom objects remain supported.

For a trusted native artifact requiring historical Lambda or NPZ behavior, use
Keras's explicit `safe_mode=False` or `serialization_lib.SafeModeScope(False)`;
these choices permit executable content and are not suitable for untrusted
models. Prefer named registered functions and embedded vocabulary assets.
The backports do not turn model factories, custom objects or Talos archives
into sandboxed inputs. An upstream Keras 2.14 native-format limitation with
compiled legacy Adam remains; HDF5 retains its optimizer continuation path.

Changing the framework distribution identity changes the recorded environment.
Preserve the old run, then start a fresh experiment or an explicit fork; do not
force an existing checkpoint to accept different dependencies.

## Validation baseline — 1 October 2026

| Environment | Result |
| --- | --- |
| Python 3.12; TensorFlow 2.21, Keras 3.15, Torch 2.14 | 159 passed |
| Python 3.11; TensorFlow 2.14.1, Keras 2.14, NumPy 1.26 | 149 passed; 10 optional-framework skips |
| Installed wheel, Python 3.10; no DL frameworks | 137 passed; 22 optional-framework skips |
| Installed wheel, Python 3.13; no DL frameworks | 137 passed; 22 optional-framework skips |
| Python 3.11; TensorFlow 2.20, Keras 3.15, Torch 2.13, Protobuf 6.33.5 | 159 passed |
| Standalone Keras using Torch, executable control documentation | 59 blocks and three provider commands passed |

The unchanged legacy training callback also passed real Iris training, prediction and Deploy/Restore on TensorFlow 2.14.1. Checks cover signal interruption, completed-trial recovery, live model/data/control edits, callable parameters, native framework artifacts in fresh processes, deleted caller sources, scientific metric references and CLI manifest lifecycle. Wheels were exercised outside the checkout; all 179 packaged source/resource files match the working sources. Lint, distribution builds and dependency consistency pass. The follow-up maintenance pass executes the documentation and all example files, raises patched dependency floors and adds cold-import and paused-Gamify regression checks; see [maintenance evidence](Maintenance.md).

This is a CPU acceptance baseline. Accelerator behavior, physical power providers and remote quantum entropy services require their own hardware/service verification. Quantum sampler adapter behavior is tested with controlled provider responses. Repeat the maintained CI matrix and archive/resume checks when updating supported dependencies.

## Read next

Use [SFD and manifest CLI](SFD_and_CLI.md) for the new entry point or
[Scan](Scan.md) to retain the existing Python workflow.
