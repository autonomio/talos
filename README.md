<h1 align="center">
  <br>
  <a href="http://autonom.io"><img src="https://raw.githubusercontent.com/autonomio/talos/master/logo.png" alt="Talos" width="350"></a>
  <br>
</h1>

<h3 align="center">Bullet-Proof Hyperparameter Experiments with TensorFlow and Keras</h3>

<p align="center">
  <a href="#talos">Talos</a> •
  <a href="#wrench-key-features">Key Features</a> •
  <a href="#arrow_forward-examples">Examples</a> •
  <a href="#floppy_disk-install">Install</a> •
  <a href="#speech_balloon-how-to-get-support">Support</a> •
  <a href="https://autonomio.github.io/talos/">Docs</a> •
  <a href="https://github.com/autonomio/talos/issues">Issues</a> •
  <a href="#page_with_curl-license">License</a> •
  <a href="https://github.com/autonomio/talos/archive/master.zip">Download</a>
</p>
<hr>
<p align="center">
Talos importantly improves ordinary TensorFlow (tf.keras) and Keras workflows  by <strong>fully automating hyperparameter experiments</strong> and <strong>model evaluation</strong>. Talos exposes TensorFlow (tf.keras) and Keras functionality entirely and there is no new syntax or templates to learn.
</p>
<p align="center">
<img src='https://i.ibb.co/3NFH646/keras-model-to-talos.gif' width=550px><br>
<em>The above animation illustrates how a minimal Sequntial model is modified for Talos</em>
</p>

### Talos

TL;DR Thousands of researchers have found Talos to importantly improve ordinary TensorFlow (tf.keras) and Keras workflows without taking away or hiding any of their power.

  - Works with ANY Keras, TensorFlow (tf.keras) or PyTorch model
  - Takes minutes to implement
  - No new syntax to learn
  - Adds zero new overhead to your workflow
  - Bullet-proof results with no breaking bugs since 2019
  - Comprehensive, up-to-date documentation

Talos is made for researchers, data scientists, and data engineers that want to remain in **complete control of their TensorFlow (tf.keras) and Keras models**, but are tired of mindless parameter hopping and confusing optimization solutions that add complexity instead of reducing it. 

<hr>

### :wrench: Key Features

**Within minutes, without learning any new syntax,** Talos allows you to configure, perform, and evaluate hyperparameter experiments that yield state-of-the-art results across a wide range of prediction tasks. Talos provides the **simplest and yet most powerful** available method for hyperparameter optimization with TensorFlow (tf.keras) and Keras. Key features include:

  - Single-line optimize-to-predict pipeline `talos.Scan(x, y, model, params).predict(x_test, y_test)`
  - Automated hyperparameter optimization
  - Model generalization evaluator
  - Experiment analytics
  - Pseudo, Quasi, and Quantum Random search options
  - Grid search
  - Probabilistic optimizers
  - Single file custom optimization strategies
  - Dynamically change optimization strategy during experiment
  - Support for man-machine cooperative optimization strategy
  - Model candidate generality evaluation
  - Live training monitor
  - Experiment analytics

Talos works on **Linux, Mac OSX**, and **Windows** systems and can be operated cpu, gpu, and multi-gpu systems.

<hr>

### :arrow_forward: Examples

Get the below code [here](https://gist.github.com/mikkokotila/4c0d6298ff0a22dc561fb387a1b4b0bb). More examples further below.

<img src=https://i.ibb.co/VWd8Bhm/Screen-Shot-2019-01-06-at-11-26-32-PM.png>

The *Simple* example below is more than enough for starting to use Talos with any Keras model. *Field Report* has +4,400 claps on Medium because it's more entertaining.

[Simple](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/A%20Very%20Short%20Introduction%20to%20Hyperparameter%20Optimization%20of%20Keras%20Models%20with%20Talos.ipynb)  [1-2 mins]

[Concise](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20on%20Keras%20with%20Breast%20Cancer%20Data.ipynb)  [~5 mins]

[Comprehensive](https://nbviewer.jupyter.org/github/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb)  [~10 mins]

[Field Report](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)  [~15 mins]

For more information on how Talos can help with your Keras, TensorFlow (tf.keras) and PyTorch workflow, visit the [User Manual](https://autonomio.github.io/talos/).

You may also want to check out a visualization of the [Talos Hyperparameter Tuning workflow](https://github.com/autonomio/talos/wiki/Workflow).

<hr>

### :floppy_disk: Install

Stable version:

#### `pip install talos`

Daily development version:

#### `pip install git+https://github.com/autonomio/talos`

<hr>

### :speech_balloon: How to get Support

| I want to...                     | Go to...                                                  |
| -------------------------------- | ---------------------------------------------------------- |
| **...troubleshoot**           | [Docs] · [Wiki] · [GitHub Issue Tracker]                   |
| **...report a bug**           | [GitHub Issue Tracker]                                     |
| **...suggest a new feature**  | [GitHub Issue Tracker]                                     |
| **...get support**            | [Stack Overflow]                     |

<hr>

### :loudspeaker: Citations

If you use Talos for published work, please cite:

`Autonomio Talos [Computer software]. (2024). Retrieved from http://github.com/autonomio/talos.`

<hr>

### :page_with_curl: License

[MIT License](https://github.com/autonomio/talos/blob/master/LICENSE)

[github issue tracker]: https://github.com/autonomio/talos/issues
[docs]: https://autonomio.github.io/talos/
[wiki]: https://github.com/autonomio/talos/wiki
[stack overflow]: https://stackoverflow.com/questions/tagged/talos
