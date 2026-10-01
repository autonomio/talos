# Talos package

`talos/` owns parameter sweeps, experiment execution, scientific records and
trained-artifact recovery. It exposes the established Python interface and the
SFD/manifest CLI over one experiment queue.

## Canonical documentation

Use the [documentation hub](../docs/README.md) to choose a workflow,
[Scan](../docs/Scan.md) for existing callbacks and
[SFD and manifest CLI](../docs/SFD_and_CLI.md) for native experiment definitions.
The [reference index](../docs/Reference/README.md) owns interface details.

## Ownership boundary

The package owns candidate scheduling, completed-trial checkpoints, parameter
and metric records, source snapshots, backend adapters and artifact selection.
It preserves the legacy callback and pandas result interface through `Scan`.

Caller code owns observations, acquisition, preparation, network definitions,
training and device determinism. No experiment reader, financial target,
indicator catalog or backtest is selected by the core. Educational dataset
helpers remain explicit opt-in calls, with their acquisition behavior described
in [Templates](../docs/Templates.md).

## Public entry points

| Surface | Public names | Responsibility |
| --- | --- | --- |
| Existing Python workflow | `talos.Scan`, `Analyze`, `Reporting`, `Predict`, `Evaluate`, `Deploy`, `Restore` | Run callbacks, inspect results, score held-out data and package/restore models |
| Native experiments | `talos.run`, `RunResult`, `UniversalExperimentLoop`, `MLManifest` | Execute caller SFDs or explicit experiment manifests |
| Post-run operations | `talos.Log`, `Sensor`, `Trainer`, `Cohort` | Inspect records, restore artifacts, explicitly retrain or select result cohorts |
| Command line | `talos` and `python -m talos` | Create, validate, execute, resume and store manifests |
| Explicit helpers | `talos.templates`, `autom8`, `callbacks`, `metrics`, `utils` | Opt-in educational models, automation and training helpers |

Exports are loaded lazily by `talos/__init__.py`. Importing the core does not
import a deep learning or plotting framework. Optional operations import their
own backend when used; choose the declared extras in
[Installation](../docs/Install_Options.md).

## Source orientation

| Path | Owns |
| --- | --- |
| `scan/`, `parameters/`, `reducers/` | Established Scan, mutable parameter facade and legacy controls |
| `experiment/` | Shared scheduling, execution, records, provenance and checkpoints |
| `sfd/`, `yaml/`, `cli/` | Templates, manifest schema/compiler/store and CLI dispatch |
| `backends/`, `commands/`, `inference/` | Framework persistence, legacy result operations and trained-artifact use |
| `preparation/`, `scalers/`, `transforms/`, `calibration/` | Explicit preparation and fitted transforms |

The adjacent `examples/` directory provides runnable caller code. The adjacent
`docs-site/` directory builds documentation; Node is not a Talos runtime
dependency. Python support and backend floors belong to
[Installation](../docs/Install_Options.md), not this orientation table.

## Operational caveats

Resume commits completed trials rather than recovering a partially trained
epoch. Archive loading executes trusted model/factory code; preserve framework
dependencies and inspect the source before restoring an unfamiliar package.
Torch persistence needs an importable factory and constructor configuration;
Keras uses native artifacts and optional custom objects. See
[Recovery](../docs/Restore.md) and [Migration](../docs/Migration.md).

## Read next

[Quickstart](../docs/Guides/Quickstart.md) provides the first runnable scan.
[Maintenance](../docs/Maintenance.md) defines compatibility and recovery proof.
