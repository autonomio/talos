<h1 align="center">
  <br>
  <a href="http://autonom.io"><img src="https://raw.githubusercontent.com/autonomio/talos/master/logo.png" alt="Talos" width="350"></a>
  <br>
</h1>

<h3 align="center">Hyperparameter Optimization for Keras</h3>

<p align="center">
  
  <a href="https://travis-ci.org/autonomio/talos">
    <img src="https://img.shields.io/travis/autonomio/talos/master.svg?style=for-the-badge&logo=appveyor" alt="Talos Travis">
  </a>

  <a href="https://coveralls.io/github/autonomio/talos">
    <img src="https://img.shields.io/coveralls/github/autonomio/talos.svg?style=for-the-badge&logo=appveyor" alt="Talos Coveralls">
  </a>

</p>

<p align="center">
  <a href="#Talos">Talos</a> •
  <a href="#Key-Features">Key Features</a> •
  <a href="#Examples">Examples</a> •
  <a href="#Install">Install</a> •
  <a href="#Install">Support</a> •
  <a href="https://autonomio.github.io/docs_talos">Docs</a> •
  <a href="https://github.com/autonomio/talos/issues">Issues</a> •
  <a href="#License">License</a> •
  <a href="https://github.com/autonomio/talos/archive/master.zip">Download</a>
</p>
<hr>
<p align="center">
Talos radically changes the ordinary Keras workflow by <strong>fully automating hyperparameter tuning</strong> and <strong>model evaluation</strong>. Talos exposes Keras functionality entirely and there is no new syntax or templates to learn.
</p>
<p align="center">
<img src='https://i.ibb.co/3NFH646/keras-model-to-talos.gif' width=550px>
</p>

### Talos

TL;DR

Talos radically transforms ordinary Keras workflows without taking away any of Keras.

- works with ANY Keras model
- takes minutes to implement
- no new syntax to learn
- adds zero new overhead to your worflow

Talos is made for data scientists and data engineers that want to remain in **complete control of their Keras models**, but are tired of mindless parameter hopping and confusing optimization solutions that add complexity instead of reducing it. Within minutes, without learning any new syntax, Talos allows you to configure, perform, and evaluate hyperparameter optimization experiments that yield state-of-the-art results across a wide range of prediction tasks. Talos provides the **simplest and yet most powerful** available method for hyperparameter optimization with Keras.

### Key Features

Based on what no doubt constitutes a "biased" review (being our own) of more than ~30 hyperparameter tuning and optimization solutions, Talos comes on top in terms of intuitive, easy-to-learn, highly permissive access to critical hyperparameter optimization capabilities. Key features include:

- Single-line optimize-to-predict pipeline `talos.Scan(x, y, model, params).predict(x_test, y_test)`
- automated hyperparameter optimization
- model generalization evaluator
- experiment analytics 
- Random search
- Grid search
- Correlation based optimization
- Pseudo, Quasi, and Quantum Random functions
- Model candidate generality evaluation
- Live training monitor
- Experiment analytics

Talos works on **Linux, Mac OSX**, and **Windows** systems and can be operated cpu, gpu, and multi-gpu systems.

### Examples

Get the below code [here](https://gist.github.com/mikkokotila/4c0d6298ff0a22dc561fb387a1b4b0bb). More examples further below.

<img src=https://i.ibb.co/VWd8Bhm/Screen-Shot-2019-01-06-at-11-26-32-PM.png>

The *Simple* example below is more than enough for starting to use Talos with any Keras model. *Field Report* has +2,600 claps on Medium because it's more entertaining.

[Simple](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/A%20Very%20Short%20Introduction%20to%20Hyperparameter%20Optimization%20of%20Keras%20Models%20with%20Talos.ipynb)  [1-2 mins]

[Concise](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20on%20Keras%20with%20Breast%20Cancer%20Data.ipynb)  [~5 mins]

[Comprehensive](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb)  [~10 mins]

[Field Report](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)  [~15 mins]

For more information on how Talos can help with your Keras workflow, visit the [User Manual](https://autonomio.github.io/docs_talos).

You may also want to check out a visualization of the [Talos Hyperparameter Tuning workflow](https://github.com/autonomio/talos/wiki/Workflow).

### Install 

Stable version:

#### `pip install talos`

Daily development version:

#### `pip install git+https://github.com/autonomio/talos.git@daily-dev`

### Support

Check out [common errors](https://github.com/autonomio/talos/wiki/Troubleshooting) in the Wiki.


Check the [Docs](https://autonomio.github.io/docs_talos) which is generally keeping up with Master (and pip package). 

If you want ask a **"how can I use Talos to..."** question, the right place is [StackOverflow](https://stackoverflow.com/questions/ask). 

If you found a bug or want to suggest a feature, check the [issues](https://github.com/autonomio/talos/issues) or [create](https://github.com/autonomio/talos/issues/new/choose) a new issue.


### License 

[MIT License](https://github.com/autonomio/talos/blob/master/LICENSE)
