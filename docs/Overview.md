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

# Workflow

The goal of a deep learning experiment is to find one or more model candidates that meet a given performance expectation. Talos makes this process as easy and streamlined as possible. Talos provides an intuitive API to manage both semi-automated and fully-automated deep learning workflows. The below example highlights a typical semi-automated workflow from idea to production ready model.

![alt text](_media/talos_deep_learning_workflow.png")

# Creating an Experiment

To get started with your first experiment is easy. You need to have three things:

a hyperparameter dictionary
a working Keras model
a Talos experiment

### STEP 1 : Prepare the input model

In order to prepare a Keras model for a Talos experiment, you simply replace parameters you want to include in the scan, with references to the parameter dictionary.

### STEP 2 : Define the parameter space

In a regular Python dictionary, you declare the hyperparameters and the boundaries you want to include in the experiment.

### STEP 3 : Configure the experiment

To start the experiment, you input the parameter dictionary and the Keras model into Talos with the option for Grid, Random, or Probabilistic optimization strategy.

### STEP 4 : Let the good times roll in

Congratulations, you've just freed yourself from the manual, laborous, and somewhat mind-numbing task of finding the right hyperparameters for your deep learning models.
