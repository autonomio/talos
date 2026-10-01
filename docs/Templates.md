# Templates Overview

Talos provides access to sets of templates consisting of the assets required by `Scan()` i.e. datasets, parameter dictionaries, and input models. In addition, `talos.templates.pipelines` consist of ready pipelines that combined the assets into a `Scan()` experiment and run it. These are mainly provided for educational, testing, and development purposes.

Each category of templates consists at least assets based on four popular machine learning datasets:

- Wisconsin Breast Cancer | [dataset info](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Cervical Cancer Screening | [dataset info](https://arxiv.org/pdf/1812.10383.pdf)
- Iris | [dataset info](https://www.semanticscholar.org/topic/Iris-flower-data-set/620769)
- Titanic Survival | [dataset info](https://www.kaggle.com/c/titanic)

In addition, some categories (e.g. datasets) include additional templates. These are listed below and can be accessed through the corresponding namespace without previous knowledge.

<hr>

# Datasets

Dataset conveniences are explicitly requested outside the experiment core. Iris is bundled with scikit-learn; the other acquisition helpers may download source data. Inputs are preprocessed for the corresponding templates. Datasets are accessed through `talos.templates.datasets`. For example:

```python
import talos
from sklearn.model_selection import train_test_split
x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=17)
assert x.shape == (150, 4) and y.shape == (150, 3)
```

#### Available Datasets

- breast_cancer
- cervical_cancer
- icu_mortality
- telco_churn
- titanic
- iris
- mnist

<hr>

# Params

Params consist of an indicative and somewhat meaningful parameter space boundaries that can be used as the parameter dictionary for `Scan()` experiments. Parameter dictionaries are accessed through `talos.templates.params`. For example:

```python
p = talos.templates.params.iris()
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'losses': ['categorical_crossentropy']})
assert 'optimizer' in p
```

#### Available Params

- breast_cancer
- cervical_cancer
- titanic
- iris

<hr>

# Models

Models consist of Keras models that can be used as an input model for `Scan()` experiments. Models are accessed through `talos.templates.models`. For example:

```python
values = {name: choices[0] for name, choices in p.items()}
history, model = talos.templates.models.iris(x_train, y_train, x_val, y_val, values)
assert len(history.history['loss']) == 2
```

#### Available Models

- breast_cancer
- cervical_cancer
- titanic
- iris

<hr>

# Pipelines

Pipelines are self-contained `Scan()` experiments where you simply execute the command and an experiment is performed. Pipelines are accessed through `talos.templates.pipelines`. For example:

```python
scan_object = talos.templates.pipelines.iris(round_limit=1)
assert len(scan_object.data) == 1
```

#### Available Pipelines

- breast_cancer
- cervical_cancer
- titanic
- iris
