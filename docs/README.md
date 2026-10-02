# Talos documentation

Talos runs parameter sweeps for Keras, TensorFlow and PyTorch models. Keep the established Python `Scan` interface or describe an experiment in a single-file definition (SFD) and run it through Python or the CLI. Both paths use the same experiment core and recorded run artifacts.

## Start with your task

| I want to… | Start here |
|---|---|
| Run my first sweep | [First parameter sweep](Guides/Quickstart.md), then the [typical example](Examples_Typical.md) |
| Keep an existing Talos model working | [Migration](Migration.md), [Scan](Scan.md), and [backend contracts](Backends.md) |
| Define a reproducible experiment for Python or the CLI | [SFD and CLI](SFD_and_CLI.md) |
| Use PyTorch, multiple inputs, multiple outputs or a generator | [Model recipes](Guides/README.md#model-recipes) |
| Reduce or control a running search | [Optimization strategies](Optimization_Strategies.md), [local strategy](Local_Strategy.md), and [Gamify](Gamify.md) |
| Compare, evaluate or reuse model candidates | [Workflow](Workflow.md), [Analyze](Analyze.md), [Evaluate](Evaluate.md), and [Predict](Predict.md) |
| Package or restore a trained model | [Deploy](Deploy.md) and [Restore](Restore.md) |
| Maintain Talos or its documentation | [Developer home](Developer/README.md) and [maintenance verification](Maintenance.md) |

## From model to recorded result

1. Choose a supported [backend environment](Backends.md). The core owns sweep execution; your framework owns model construction, training and device behavior.
2. Supply data and a working training callback to [Scan](Scan.md), or place the parameter, preparation and model functions in an [SFD](SFD_and_CLI.md). Data loading and scientific preprocessing remain experiment code.
3. Declare the candidate values and [search policy](Optimization_Strategies.md). The parameter domain selects pending combinations; configured reducers or live controls may remove later candidates.
4. The experiment runner prepares and trains each selected candidate. Backend adapters normalize its history and retain the trained model according to the [backend contract](Backends.md).
5. Inspect the first observable result in `scan.data` or the native run result, then open the run directory’s `results.csv`. [SFD and CLI](SFD_and_CLI.md) describes manifests, source snapshots, round records, checkpoints and resume validation.
6. [Analyze](Analyze.md) tuning results, [evaluate](Evaluate.md) selected candidates on held-out data, and [predict](Predict.md) or [archive](Deploy.md) a chosen model. Talos ends at recorded experiments and reusable model assets; application serving and scientific conclusions remain yours.

The [workflow guide](Workflow.md) connects these steps and explains when to repeat the search.

## Documentation map

| Section | Canonical responsibility | Entry point |
|---|---|---|
| Overview | Product boundary, capabilities, workflow and direction | [Capabilities](Overview.md) and [development priorities](Roadmap.md) |
| Guides | Complete reader jobs, model recipes and operating a sweep | [Guides](Guides/README.md) |
| Reference | Public interfaces, defaults, return values and error boundaries | [Reference](Reference/README.md) |
| Developer | Contributions, verification, documentation and maintenance | [Developer](Developer/README.md) |
| Packages | Source ownership, public entry points and optional dependencies | [Talos package](../talos/README.md) |

## Scope and responsibility

Talos owns parameter selection, experiment execution, result recording, reducer and intervention controls, checkpoint validation and supported model archive contracts. It accepts caller-provided arrays and preparation functions; it does not supply a finance data pipeline, decide whether a metric is scientifically suitable, establish causal validity or deploy an application server.

A seed controls supported sampling and recorded RNG state. Hardware, framework kernels, external services and opaque data streams can impose further reproducibility limits. Review [backend compatibility](Backends.md), [recovery contracts](SFD_and_CLI.md) and [maintenance evidence](Maintenance.md) for the claims proven in each environment.

## Read next

Start with the [first sweep](Guides/Quickstart.md) for the Python interface, [SFD and CLI](SFD_and_CLI.md) for a file-based experiment, or [migration](Migration.md) for an existing Talos project.

## Cite an experiment

Use the [citation guide](Citing_Talos.md) to cite the software version and retain manifest and run identities with research artifacts.
