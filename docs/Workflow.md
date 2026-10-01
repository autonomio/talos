# Model-search workflow

The goal of a deep learning experiment is to find one or more model candidates that meet a given performance expectation. Talos provides an API for both semi-automated and fully automated workflows. This guide connects the established path from an experiment idea to a model archive.

The [original workflow illustration](_media/talos_deep_learning_workflow.png) remains available as historical context. The procedure below describes current model evaluation, archiving and reuse.

## Prerequisites

Have a supported [backend](Backends.md), a working training callback, candidate parameter values and a writable experiment directory. Decide which metric expresses the research question, which direction is better and which data will remain untouched until evaluation. The [first parameter sweep](Guides/Quickstart.md) supplies a bounded starting example.

## Procedure

1. Prepare the data and model. Fit preprocessing on training data, use a separate validation split for tuning and retain a final test split. Talos receives this prepared data; it does not determine whether the split represents the deployment population.
2. Declare the parameter space and run [Scan](Scan.md). Use the [typical example](Examples_Typical.md) for the callback interface or [SFD and CLI](SFD_and_CLI.md) for an importable experiment file.
3. Inspect completed trials with [Analyze](Analyze.md). Compare metric distributions and parameter choices before increasing the search budget.
4. Decide whether to revise the experiment. Change the candidate space or [optimization policy](Optimization_Strategies.md) for another run; use [local controls](Local_Strategy.md) when a documented live intervention is appropriate.
5. [Evaluate](Evaluate.md) selected candidates on held-out data. Validation performance drove tuning and cannot serve as independent evidence of generalization.
6. [Predict](Predict.md) with a selected candidate, then [Deploy](Deploy.md) it to a model archive and [Restore](Restore.md) it in a compatible environment.

## Observable result

A successful scan returns a results table and run directory containing trial records, metadata, source snapshots and checkpoints. Analysis summarizes that table. Evaluation produces held-out measurements. Prediction produces outputs for supplied inputs. Deployment writes a zip archive, and restoration exposes the supported model and stored experiment assets.

These are distinct outcomes: completing a sweep does not guarantee the target performance, and restoring a model does not create a running production service.

## Failure boundaries

- Check the [callback and backend contract](Backends.md) when training returns the wrong object or metrics.
- Check the data split, leakage and metric definition when a model appears unexpectedly strong or weak.
- Check the [recovery contract](SFD_and_CLI.md) when a resumed run has changed source, data or configuration.
- Check [Restore](Restore.md) for framework compatibility, custom objects and Torch factory reconstruction when moving an archive.

## Read next

Use [Analyze](Analyze.md) after the first scan, [Evaluate](Evaluate.md) before accepting a candidate, and [maintenance verification](Maintenance.md) before changing the framework environment that runs it.
