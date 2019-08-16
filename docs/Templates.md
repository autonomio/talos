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

Datasets are preprocessed so that they can be used directly as inputs for deep learning models. Datasets are accessed through `talos.templates.datasets`. For example:

```python
talos.templates.datasets.breast_cancer()

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
talos.templates.params.breast_cancer()
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
talos.templates.models.breast_cancer()
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
scan_object = talos.templates.pipelines.breast_cancer()
```

#### Available Pipelines

- breast_cancer
- cervical_cancer
- titanic
- iris
