# Templates

`talos.templates` exposes dataset acquisition helpers, preset parameter dictionaries, model callbacks and complete Scan pipelines. Import it through `talos.templates`. These existing Python templates support education, testing and development; framework-specific [SFD templates](SFD_and_CLI.md) separately support CLI experiments.

Dataset acquisition is explicitly invoked by the caller. Dataset helpers read remote Autonomio CSVs, except MNIST, which uses the TensorFlow dataset loader and its download/cache behavior. Install [TensorFlow](Install_Options.md) for the preset parameter dictionaries, model callbacks and pipelines; dataset CSV helpers use the Talos core and network access.

Each category of templates consists at least assets based on four popular machine learning datasets:

| - Wisconsin Breast Cancer | [dataset info](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) |
| - Cervical Cancer Screening | [dataset info](https://arxiv.org/pdf/1812.10383.pdf) |
| - Iris | [dataset info](https://archive.ics.uci.edu/dataset/53/iris) |
| - Titanic Survival | [dataset info](https://www.kaggle.com/competitions/titanic) |

In addition, some categories (e.g. datasets) include additional templates. These are listed below and can be accessed through the corresponding namespace without previous knowledge.

<hr>

## Datasets

Dataset conveniences are explicitly requested outside the experiment core. `iris()` downloads the Autonomio CSV, shuffles its rows and returns features with one-hot labels. Use scikit-learn's separate `load_iris` fixture in [Scan](Scan.md#minimal-example) when a bundled, network-independent dataset is required. Datasets are accessed through `talos.templates.datasets`. For example:

```python
import talos
from sklearn.model_selection import train_test_split
x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=17)
assert x.shape == (150, 4) and y.shape == (150, 3)
```

### Available datasets

- breast_cancer
- cervical_cancer
- icu_mortality
- telco_churn
- titanic
- iris
- mnist

<hr>

## Parameter dictionaries

Params consist of an indicative and somewhat meaningful parameter space boundaries that can be used as the parameter dictionary for `Scan()` experiments. Parameter dictionaries are accessed through `talos.templates.params`. For example:

```python
p = talos.templates.params.iris()
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'losses': ['categorical_crossentropy']})
assert 'optimizer' in p
```

### Available parameter dictionaries

- breast_cancer
- cervical_cancer
- titanic
- iris

<hr>

## Model callbacks

Models consist of Keras models that can be used as an input model for `Scan()` experiments. Models are accessed through `talos.templates.models`. For example:

```python
values = {name: choices[0] for name, choices in p.items()}
history, model = talos.templates.models.iris(x_train, y_train, x_val, y_val, values)
assert len(history.history['loss']) == 2
```

### Available model callbacks

- breast_cancer
- cervical_cancer
- titanic
- iris

<hr>

## Complete pipelines

Pipelines are self-contained `Scan()` experiments where you simply execute the command and an experiment is performed. Pipelines are accessed through `talos.templates.pipelines`. For example:

```python
scan_object = talos.templates.pipelines.iris(round_limit=1)
assert len(scan_object.data) == 1
```

### Available pipelines

- breast_cancer
- cervical_cancer
- titanic
- iris

## Return contracts and execution boundaries

| Namespace | Public calls and outputs |
| --- | --- |
| `datasets` | `iris()`, `breast_cancer()`, `cervical_cancer()`, `titanic()`, `icu_mortality(samples=None)`, and `telco_churn(quantile=.5)` return `(x, y)`; Telco returns a list of two targets. |
| `datasets.mnist()` | Returns `(x_train, y_train, x_val, y_val)` with scaled image inputs and one-hot labels; prints the chosen input shape. |
| `params` | `iris()`, `breast_cancer()`, `cervical_cancer()` and `titanic(debug=False)` return candidate dictionaries with lists/range tuples and TensorFlow optimizer classes. |
| `models` | Each named model accepts `(x_train, y_train, x_val, y_val, params)` with scalar per-trial values and returns `(history, fitted_model)`. Calling it performs training. |
| `pipelines` | Each named pipeline accepts `round_limit=2, random_method='uniform_mersenne'`; Titanic also accepts `debug=False`. Calling it acquires data and runs Scan immediately. |

Pipelines use the experiment name `'test'` and create a new run folder. Even a one-trial pipeline can train for the preset epoch count; reduce epochs in an explicit parameter dictionary when a shorter fit is required. The page's Iris callback example uses an explicit two-epoch dictionary, while the pipeline example uses its own presets.

CSV helper sampling and row shuffling are not seeded through these function signatures. Titanic retains missing values and reports that condition. Telco's positive quantile transforms its two result targets into binary labels; a nonpositive quantile retains continuous values. Helpers do not establish train-only preprocessing or a final held-out scientific evaluation protocol for the caller.

Remote file availability, schema changes, missing values, filesystem/download failures and framework errors propagate. Do not treat preset architecture/parameter choices as validated model recommendations. Native Torch models require a caller-owned callback or the [Torch SFD](../examples/sfd/torch_sfd.py).

## Read next

Use the [typical experiment walkthrough](Examples_Typical.md) to assemble your own assets, [Scan](Scan.md#minimal-example) for a bounded offline Iris fixture, or [SFD and CLI](SFD_and_CLI.md) for framework templates and manifest-controlled execution.
