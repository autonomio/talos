# Guides

These guides take a model from its first parameter sweep to recorded results, controlled search and reusable model assets. They preserve the established Talos callback workflow alongside the SFD and CLI path.

## Before you begin

Use a supported Python and [backend installation](../Install_Options.md), a writable experiment directory and a working training function. Each model recipe states its own dataset, framework and execution order. The bundled datasets used below are offline fixtures; a short recipe verifies an interface rather than the performance of a research model.

## First sweep and migration

- [First parameter sweep](Quickstart.md): compare a Keras model before and after adding Talos.
- [Workflow](../Workflow.md): decide when to revise, evaluate or archive a search.
- [Migration](../Migration.md): keep current model callbacks and adopt corrected Talos 2 contracts.
- [SFD and CLI](../SFD_and_CLI.md): run file-based experiments, inspect artifacts and resume checkpoints.

## Model recipes

| Model shape or task | Walkthrough | Complete runnable code |
|---|---|---|
| Typical Keras classification | [Iris sweep](../Examples_Typical.md) | [Complete Iris example](../Examples_Typical_Code.md) |
| Multiple inputs | [Aligned input arrays](../Examples_Multiple_Inputs.md) | [Complete multiple-input example](../Examples_Multiple_Inputs_Code.md) |
| Multiple outputs | [Diagnosis and radius targets](../Examples_Multiple_Outputs.md) | [Complete multiple-output example](../Examples_Multiple_Outputs_Code.md) |
| Sequence generator | [Handwritten digits](../Examples_Generator.md) | [Complete generator example](../Examples_Generator_Code.md) |
| Bounded AutoML | [Binary Iris search](../Examples_AutoML.md) | [Complete AutoML example](../Examples_AutoML_Code.md) |
| Native PyTorch | [Training history and factories](../Examples_PyTorch.md) | [Complete PyTorch example](../Examples_PyTorch_Code.md) |

The walkthroughs explain each step. Their complete-code companions are the corresponding standalone programs; use the walkthrough as the authority for prerequisites, data boundaries and interpretation.

## Operate the search

1. Choose a [grid or sampled search](../Optimization_Strategies.md).
2. Configure [probabilistic reduction](../Probabilistic_Reduction.md) or a [custom reducer](../Custom_Reducers.md) when the completed metrics support that decision.
3. Use a [local strategy](../Local_Strategy.md) or [Gamify](../Gamify.md) to change pending work during a run.
4. Configure [devices and worker shards](../Parallelism.md) for the environment where the callback executes.

A completed trial produces result rows and run artifacts. Reducers affect pending trials; device helpers configure a framework and do not launch concurrent workers. Each operational guide names the controls and limits it owns.

## When a workflow fails

Check the [backend contract](../Backends.md) before changing the search policy. Check [Scan](../Scan.md) for callback arguments, data alignment and returned objects; check [SFD and CLI](../SFD_and_CLI.md) for checkpoint identity and recovery errors. A reproducible bug report belongs in [support](../Asking_Help.md).

## Read next

Run the [first sweep](Quickstart.md), or choose the recipe matching your current model. After training, continue with [Analyze](../Analyze.md), [Evaluate](../Evaluate.md) and [Deploy](../Deploy.md).

Use the [citation guide](../Citing_Talos.md) to identify the software and recorded experiment in a research paper.
