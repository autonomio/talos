# Key Features

Talos features focus on easy-of-use, intuitive workflow, do not introduce any new syntax, or require reading through complex technical documentations.

- perform a hyperparameter scan
- perform a model architecture search
- analyze and visualize results
- evaluate results to find best model candidates
- make predictions with selected models
- deploy models into self-contained zip files
- restore models onto a production system
- create AutoML pipelines

If you have an idea for a new feature, don't hesitate to [suggest it](https://github.com/autonomio/talos/issues/new).

# How to Use

Talos provides single-line commands for conducting and analyzing experiments, for evaluating models from the experiment, and making predictions with models. In addition, Talos provides s streamlined way to deploy and restore models across systems.

The primary commands return a class object specific to the command, with various properties. These are outlined in the corresponding sections of the documentation.

`Analyze()`, `Evaluate()`, `Predict()`, and `Deploy()` accept the class object resulting from `Scan()` as input; `Restore()` accepts an archive path.


#### conduct an experiment

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
scan_object = talos.Scan(x, y, p, input_model, 'overview',
                         x_val=x_val, y_val=y_val, seed=17, disable_progress_bar=True)
```

#### analyze the results of an experiment

```python
talos.Analyze(scan_object)
```

#### evaluate one or more models from the experiment

```python
talos.Evaluate(scan_object)
```

#### make predictions with a model from the experiment

```python
talos.Predict(scan_object)
```

#### create a deploy package

```python
talos.Deploy(scan_object, model_name='deployed_package', metric='val_loss', asc=True)
```

#### deploy a model

```python
talos.Restore('deployed_package.zip')
```

In addition to the primary commands, various utilities can be accessed through `talos.utils`, datasets, parameters, and models from `talos.templates`, and AutoML features through `talos.autom8`.


# Creating an Experiment

To get started with your first experiment is easy. You need to have three things:

a hyperparameter dictionary
a working Keras, tf.keras or Torch training callback
a Talos experiment

### STEP 1 : Prepare the input model

In order to prepare a Keras model for a Talos experiment, you simply replace parameters you want to include in the scan, with references to the parameter dictionary.

### STEP 2 : Define the parameter space

In a regular Python dictionary, you declare the hyperparameters and the boundaries you want to include in the experiment.

### STEP 3 : Configure the experiment

To start the experiment, you input the parameter dictionary and the Keras model into Talos with the option for Grid, Random, or Probabilistic optimization strategy.

### STEP 4 : Let the good times roll in

Congratulations, you've just freed yourself from the manual, laborous, and somewhat mind-numbing task of finding the right hyperparameters for your deep learning models.
