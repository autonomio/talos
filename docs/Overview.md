# Talos capabilities

## Key features

Talos organizes model search around a training callback, a parameter dictionary and reusable result objects. The Python workflow retains the model code you already use; SFD and CLI execution add a file-based experiment definition.

- perform a hyperparameter scan
- perform a model architecture search
- analyze and visualize results
- evaluate results to find best model candidates
- make predictions with selected models
- archive supported models and run assets in zip files
- restore models in a compatible runtime
- create AutoML pipelines

Propose a feature in the [issue tracker](https://github.com/autonomio/talos/issues/new).

## How to use

Before running these examples, install the [backend](Backends.md) used by the callback and execute the linked Scan setup. The working directory must be writable because scans and archives create files.

Talos provides commands for conducting and analyzing experiments, evaluating candidates and making predictions. Archive commands transfer supported model assets between compatible systems; they do not start an application server.

The primary commands return a class object specific to the command, with various properties. These are outlined in the corresponding sections of the documentation.

`Analyze()`, `Evaluate()`, `Predict()`, and `Deploy()` accept the class object resulting from `Scan()` as input; `Restore()` accepts an archive path.

### Conduct an experiment

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
scan_object = talos.Scan(x, y, p, input_model, 'overview',
                         x_val=x_val, y_val=y_val, seed=17, disable_progress_bar=True)
```

### Analyze experiment results

```python
talos.Analyze(scan_object)
```

### Prepare held-out evaluation

```python
talos.Evaluate(scan_object)
```

### Prepare predictions

```python
talos.Predict(scan_object)
```

### Create an archive

```python
talos.Deploy(scan_object, model_name='deployed_package', metric='val_loss', asc=True)
```

### Restore the archive

```python
talos.Restore('deployed_package.zip')
```

In addition to the primary commands, various utilities can be accessed through `talos.utils`, datasets, parameters, and models from `talos.templates`, and AutoML features through `talos.autom8`.

## Creating an experiment

An experiment needs three inputs:

- a hyperparameter dictionary
- a working Keras, tf.keras or Torch training callback
- a Talos experiment configuration

### 1. Prepare the input model

In order to prepare a Keras model for a Talos experiment, you simply replace parameters you want to include in the scan, with references to the parameter dictionary.

### 2. Define the parameter space

In a regular Python dictionary, you declare the hyperparameters and the boundaries you want to include in the experiment.

### 3. Configure the experiment

To start the experiment, you input the parameter dictionary and the Keras model into Talos with the option for Grid, Random, or Probabilistic optimization strategy.

### 4. Inspect the completed trials

The completed scan exposes trial metrics and selected parameter values in `scan_object.data`; the run directory retains CSV results and recovery artifacts. Use those observations to decide whether to revise the search or evaluate a selected candidate on untouched data.

## Outputs and failure boundaries

`Analyze`, `Evaluate` and `Predict` construct helpers around the completed run; constructing them does not itself run held-out evaluation or prediction. Use their reference methods for those operations. `Deploy` writes `deployed_package.zip`; `Restore` loads that archive using the matching framework and, when required, an importable model factory.

Data loading, preprocessing, training and scientific interpretation remain caller responsibilities. A missing metric, incompatible model input shape or unavailable archive factory needs correction at that boundary; changing the search policy will not repair it. See [Scan](Scan.md), [Backends](Backends.md) and [Restore](Restore.md).

## Read next

Run the [first parameter sweep](Guides/Quickstart.md), follow the [workflow](Workflow.md), or move an existing callback to [SFD and CLI](SFD_and_CLI.md).
