# SFD and manifest CLI

An SFD defines `params()`, `prep(data, round_params)` and `model(prepared, round_params)` in ordinary Python. Training remains your code. Model returns accept `(history, model)` or a dictionary containing numeric metrics plus `model`, `history`, optional `predictions`, `backend` and `factory`.

## Run from Python

```python
from talos import run

result = run('examples/sfd/tensorflow_sfd.py', data=my_splits,
             experiment_name='iris', seed=42,
             objective={'metric': 'val_loss', 'direction': 'min'})
result.predict(x_test)
```

`performance_target=['val_loss', 0.2, True]` stops once a metric reaches its threshold (`True` minimizes, `False` maximizes); it works from Python or `uel.performance_target`. Resuming an achieved target does not train additional trials.

`prep` can accept no arguments, caller data only, or data and parameters. Opaque data is passed unchanged. The real Iris SFD examples can also load their own fixture when called without data; project scaffolds require you to implement loading or supply data.

An SFD may alternatively expose `params()` and a `manifest()` factory (or a manifest object) with `prepare_data` and `run_model`. The compiler and Python executor use the same contract.

Use `talos.experiment.UniversalExperimentLoop` for the Limen-style Python facade. Generic `talos.experiment.manifest_core.MLManifest` supplies optional fluent preparation, train-only fitted transforms/scaling/PCA, explicit targets/splits and calibration. It receives caller data and defines no readers or financial targets.

## Create and validate

```sh
talos new my-study
cd my-study
talos init first --template tf_keras
# Edit manifests/first_sfd.py: implement prep using your own data.
talos validate manifests/first.yaml
talos run --dry-run manifests/first.yaml
```

Templates are `keras`, `tf_keras` and `pytorch`. The generated YAML points at the copied editable Python file. Validation checks schema; dry-run also resolves Python and parameter/pruner references without training. Import-time behavior in your module remains your responsibility.

## Manifest structure

```yaml
schema_version: "1.0"
metadata:
  name: first
  mode: development
sfd:
  module: first_sfd.py
  backend: tensorflow
  objective:
    metric: val_loss
    direction: min
  params:
    epochs: [5, 10]
uel:
  seed: 42
  round_limit: 4
  search_strategy:
    type: grid
  checkpoint_interval: 1
  save_models: true
  output_format: csv
```

`module` can be an importable module or a project `.py` file. Parameter overrides merge with `params()`. Literal strings stay literal; explicit callable values use `{callable: 'package.module:qualified_name'}`. Optional `sfd.context` carries plain caller configuration to `prep`. No `data_source`, finance configuration, feature catalog or reader registry exists.

Grid search is lazy; random search has a finite legal domain and reproducible queue state. Both allow runtime domain mutation and priority injections. Legacy `Scan` uses its own ordering/sampling facade over the same mutable queue. General pruning strategies are `correlation`, `sanity`, `saturation`, `budget` and `focus`; consult their Python constructors for parameters.

## Execute and control

```sh
talos run --no-progress-bar manifests/first.yaml
talos run --resume results/dev/first_<timestamp>
```

Development writes `results/dev/`; production writes `results/`. `uel.output_path` accepts `{name}` and `{datetime}`. `output_format: parquet` additionally writes Parquet with Polars, without requiring Arrow. Python runs provide `experiment_dir` directly and support `output_format='parquet'`. Mixed typed parameter categories use canonical encodings in the Parquet/pruning view; `.data` and callbacks retain their live Python values.

SIGINT/SIGTERM pause the run and checkpoint completed trials. `stop_after` or a result's `request_pause()` supports controlled Python pauses. Resume validates the recorded manifest and run identity, using verified source bundles when original caller files disappear; surviving originals must still match their recorded hashes. See [migration](Migration.md) for completed-trial recovery and data fingerprints.

Write `interventions.json` in the run directory to control the next queued trial:

```json
[{"op": "keep_between", "param": "learning_rate", "lower": 0.001, "upper": 0.01}]
```

Other operations include `remove_is`, `remove_ge`, `remove_le`, `keep_is`, `inject`, `inject_value`, `trim`, `set_filter` and `clear_filter`. Suggestions and applied interventions are distinguished in the audit. Python intra callbacks receive the experiment log and mutable queue at `feedback_interval` completed trials.

## Store and provenance

```sh
talos commit manifests/first.yaml -m 'initial study'
talos ls
talos run manifest://sha256:<manifest-hash>
talos fork sha256:<manifest-hash> second
talos lineage sha256:<manifest-hash>
talos reindex
talos backup
```

Committed manifests are immutable and content-addressed; full or unambiguous short hashes resolve to the same identity. Fork records its parent, lineage prints ancestry, and reindex rebuilds the derived store index. Backup commits/pushes the project to its configured Git remote; restore with `talos new restored --from <remote>`.

`profile` reports domain complexity and samples caller model runs to estimate runtime. It executes your code and reports sample errors. Profiles do not acquire data independently.

Inspect saved results with `RunResult.load`, `talos.log.Log`, `talos.inference.Sensor` and `talos.cohort.Cohort`. Sensor restores trained artifacts. Explicit `Trainer` operations retrain selected parameters when requested; restoration never silently retrains.
