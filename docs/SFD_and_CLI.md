# SFD and manifest CLI

An SFD defines `params()`, `prep(data, round_params)` and `model(prepared, round_params)` in ordinary Python. Training remains your code. Model returns accept `(history, model)` or a dictionary containing numeric metrics plus `model`, `history`, optional `predictions`, `backend` and `factory`.

## Prerequisites and scope

Install the backend extra used by the SFD; the examples below use
`talos[tensorflow]`. Python examples run from the repository checkout where
`examples/sfd/` exists. The CLI project example uses your working directory and
creates a local Git-backed study; backup later writes to its configured remote.
See [Installation](Install_Options.md) for supported Python and framework versions.

This guide follows preparation, validation, execution, recovery and manifest
storage. [Scan](Scan.md) remains the established callback interface.

## Run from Python

Run the next block from the repository root; the SFD loads the bundled Iris fixture in its own `prep`.

```python
from talos import run

result = run('examples/sfd/tensorflow_sfd.py',
             experiment_name='iris', seed=42,
             objective={'metric': 'val_loss', 'direction': 'min'})
from examples.sfd.tensorflow_sfd import prep
x_test = prep(None, {})['x_val']
predictions = result.predict(x_test)
```

`performance_target=['val_loss', 0.2, True]` stops once a metric reaches its threshold (`True` minimizes, `False` maximizes); it works from Python or `uel.performance_target`. Resuming an achieved target does not train additional trials.

`prep` can accept no arguments, caller data only, or data and parameters. Opaque data is passed unchanged. The real Iris SFD examples can also load their own fixture when called without data; project scaffolds require you to implement loading or supply data.

An SFD may alternatively expose `params()` and a `manifest()` factory (or a manifest object) with `prepare_data` and `run_model`. The compiler and Python executor use the same contract.

Use `talos.experiment.UniversalExperimentLoop` for the native Python facade. Generic `talos.experiment.manifest_core.MLManifest` supplies optional fluent preparation, train-only fitted transforms/scaling/PCA, explicit targets/splits and calibration. It receives caller data and defines no readers or financial targets.

## Create and validate

```sh
talos new my-study
cd my-study
talos init first --template tf_keras
# Edit manifests/tf_keras_sfd.py: implement prep using your own data.
talos validate manifests/first.yaml
talos run --dry-run manifests/first.yaml
```

Templates are `keras`, `tf_keras` and `pytorch`. The generated YAML points at the copied editable Python file. Validation checks schema; dry-run also resolves Python and parameter/pruner references without training. Import-time behavior in your module remains your responsibility.

For a complete runnable Iris study, save the following caller-owned module as `manifests/first_sfd.py` and replace `manifests/first.yaml` with the YAML shown below. The fixture loader is explicitly inside your `prep` function.

```python
# Save as manifests/first_sfd.py inside my-study.
backend = 'tensorflow'


def params():
    return {'neurons': [8, 16], 'learning_rate': [.001, .01],
            'epochs': [1], 'batch_size': [16]}


def prep(data, round_params):
    if data is not None:
        return data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    x, y = load_iris(return_X_y=True)
    xt, xv, yt, yv = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
    return {'x_train': xt, 'x_val': xv, 'y_train': yt, 'y_val': yv}


def model(data, round_params):
    from tensorflow import keras
    network = keras.Sequential([keras.layers.Input((4,)),
                                keras.layers.Dense(round_params['neurons'], activation='relu'),
                                keras.layers.Dense(3, activation='softmax')])
    network.compile(optimizer=keras.optimizers.Adam(round_params['learning_rate']),
                    loss='sparse_categorical_crossentropy')
    history = network.fit(data['x_train'], data['y_train'],
                          validation_data=(data['x_val'], data['y_val']),
                          epochs=round_params['epochs'],
                          batch_size=round_params['batch_size'], verbose=0)
    return history, network
```

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
    epochs: [1, 2]
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

Objective strings such as `loss` retain metric inference. An objective mapping accepts `direction: min` or `direction: max`; other directions fail validation before caller imports. `save_models`, `prep_each_round` and `progress_bar` require YAML booleans (`true` or `false`), so a quoted `"false"` cannot silently enable a policy. These checks apply to both direct compilation and saved-manifest recovery.

Grid search is lazy; random search has a finite legal domain and reproducible queue state. Both allow runtime domain mutation and priority injections. Legacy `Scan` uses its own ordering/sampling facade over the same mutable queue. General pruning strategies are `correlation`, `sanity`, `saturation`, `budget` and `focus`; consult their Python constructors for parameters.

## Execute and control

```sh
talos run --no-progress-bar manifests/first.yaml
run_dir=$(python -c "from pathlib import Path; print(max(Path('results/dev').glob('first_*'), key=lambda p: p.stat().st_mtime))")
talos run --resume "$run_dir"
```

Development writes `results/dev/`; production writes `results/`. `uel.output_path` accepts `{name}` and `{datetime}`. `output_format: parquet` additionally writes Parquet with Polars, without requiring Arrow. Python runs provide `experiment_dir` directly and support `output_format='parquet'`. Mixed typed parameter categories use canonical encodings in the Parquet/pruning view; `.data` and callbacks retain their live Python values.

SIGINT/SIGTERM pause the run and checkpoint completed trials. `stop_after` or a result's `request_pause()` supports controlled Python pauses. Resume validates the recorded manifest and run identity, using verified source bundles when original caller files disappear; surviving originals must still match their recorded hashes. See [migration](Migration.md) for completed-trial recovery and data fingerprints.

Write `interventions.json` in the run directory to control the next queued trial:

```json
[{"op": "keep_between", "param": "learning_rate", "lower": 0.001, "upper": 0.01}]
```

Other operations include `remove_is`, `remove_ge`, `remove_le`, `keep_is`, `inject`, `inject_value`, `trim`, `set_filter` and `clear_filter`. Suggestions and applied interventions are distinguished in the audit. Python intra callbacks receive the experiment log and mutable queue at `feedback_interval` completed trials.

## Store and provenance

Before committing, change `metadata.mode` in `manifests/first.yaml` to `production`. Commit accepts production manifests. Replace `MANIFEST_ID` below with the full hash printed by `commit`. The `MANIFEST_ID=...` assignment is a placeholder for that printed value. For the backup example, configure a local test remote with `git init --bare --initial-branch=main ../study-backup.git` and `git remote add origin ../study-backup.git`. Talos projects start on `main`; the bare remote must point its default branch there for clone restoration. In the existing `[store]` section of `talos.toml`, set `backup_remote = "../study-backup.git"`; backup reads this setting.

```sh
talos commit manifests/first.yaml -m 'initial study'
MANIFEST_ID=sha256:<manifest-hash>
talos ls
talos run "manifest://${MANIFEST_ID}"
talos fork "$MANIFEST_ID" second
talos lineage "$MANIFEST_ID"
talos reindex
talos backup
```

Committed manifests are immutable and content-addressed; full or unambiguous short hashes resolve to the same identity. Resolution checks both `lineage.id` and the recomputed content digest against the stored filename. Recommit and fork reject a changed committed copy; reindex reports and excludes it from the derived index. The lineage envelope remains outside the content digest, preserving historical commit and runtime identities.

Working YAML files remain mutable. Editing a draft produces a new content identity at the next commit, even when the draft retains an earlier `lineage.id`. Fork records its parent and lineage prints ancestry. Backup commits/pushes the project to its configured Git remote; restore with `talos new restored --from <remote>`.

`profile` reports domain complexity and samples caller model runs to estimate runtime. It executes your code and reports sample errors. Profiles do not acquire data independently.

Inspect saved results with `RunResult.load`, `talos.log.Log`, `talos.inference.Sensor` and `talos.cohort.Cohort`. Sensor restores trained artifacts. Explicit `Trainer` operations retrain selected parameters when requested; restoration never silently retrains.

## Expected outputs and failure boundaries

The Python example returns a `RunResult` with four completed combinations and
predictions for its held-out Iris split. The CLI example creates a study, an
editable SFD and manifest, then writes completed-trial records and trained
artifacts under the printed run directory. `validate` and `--dry-run` report
configuration or resolution errors before execution; they do not prove training
will succeed.

Training failures, missing backend dependencies and user preparation errors
belong to the caller model or environment. Resume rejects changed identities,
source or verified data; restore the recorded inputs before retrying. A pause
commits completed trials and does not checkpoint an unfinished training epoch.
`commit` requires production mode; backup requires a usable configured Git remote.
The local bare remote above provides a bounded test of that write.

## Read next

[Migration](Migration.md) wraps existing callbacks. [Restore](Restore.md) covers
trained-artifact use; [Maintenance](Maintenance.md) defines portability checks.
