# Native PyTorch sweep

Train two PyTorch networks while retaining the established Talos callback interface. The recipe records epoch metrics explicitly and provides a reconstruction factory for native Torch archives. The [complete program](Examples_PyTorch_Code.md) guards training so archive restoration can import the model without starting another sweep.

## Prerequisites

Use Python 3.10–3.13 with the [Torch extra](Backends.md) (`talos[torch]`) installed in the active interpreter. The scikit-learn dataset is available offline through the core dependencies. Run the Python blocks in order, in one session, from a writable experiment directory. These bounded training runs demonstrate the interface; they do not establish clinical or generalization performance.

## Procedure

1. Import the libraries for this recipe.
2. Prepare aligned training and validation data.
3. Define the callback, or select the built-in AutoML model.
4. Declare the parameter candidates.
5. Run the bounded Scan configuration and inspect its completed rows.

### Imports

```python
import talos
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
```

### Loading data

```python
x, y = load_breast_cancer(return_X_y=True)
x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=144, test_size=36, stratify=y, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train = torch.as_tensor(scaler.transform(x_train), dtype=torch.float32)
x_val = torch.as_tensor(scaler.transform(x_val), dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.long)
y_val = torch.as_tensor(y_val, dtype=torch.long)
```

Unlike Keras, PyTorch expects the input data to be converted into tensors.

### Defining the model

```python
class BreastCancerNet(nn.Module, talos.utils.TorchHistory):
    def __init__(self, n_feature, first_neuron=8, second_neuron=4, dropout=.1):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_feature, first_neuron), nn.ReLU(),
                                    nn.Dropout(dropout), nn.Linear(first_neuron, second_neuron),
                                    nn.ReLU(), nn.Linear(second_neuron, 2))
        self.init_history()

    def forward(self, x):
        return self.layers(x)


def build_network(n_feature, first_neuron=8, second_neuron=4, dropout=.1):
    return BreastCancerNet(n_feature, first_neuron, second_neuron, dropout)


def breast_cancer(x_train, y_train, x_val, y_val, params):
    config = {name: params[name] for name in ('first_neuron', 'second_neuron', 'dropout')}
    config['n_feature'] = x_train.shape[1]
    net = build_network(**config)
    # An importable module-level factory makes state_dict archives portable.
    net.talos_factory, net.talos_config = build_network, config
    optimizer = getattr(torch.optim, params['optimizer'])(net.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()
    for _ in range(params['epochs']):
        net.train()
        for start in range(0, len(x_train), params['batch_size']):
            optimizer.zero_grad()
            loss = criterion(net(x_train[start:start + params['batch_size']]),
                             y_train[start:start + params['batch_size']])
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            train_logits, val_logits = net(x_train), net(x_val)
            net.append_loss(criterion(train_logits, y_train).item())
            net.append_val_loss(criterion(val_logits, y_val).item())
            net.append_metric(f1_score(y_train.numpy(), train_logits.argmax(1).numpy()))
            net.append_val_metric(f1_score(y_val.numpy(), val_logits.argmax(1).numpy()))
    # The historical Torch return is supported; (net.history, net) also works.
    return net, net.parameters()
```

In order to unify `Scan()` API for Keras and PyTorch, a Keras-like history with epoch-by-epoch metrics is required. This is achieved with `talos.utils.TorchHistory` helper:

```python
preview_net = build_network(n_feature=x_train.shape[1])
print(preview_net)  # TorchHistory is mixed into the network defined above.
print(preview_net.history)  # Epoch metrics are initially empty.
```

In each iteration (epoch), metrics must be computed and then appended to the history object:

```python
preview_params = {'first_neuron': 8, 'second_neuron': 4, 'dropout': .1,
                  'optimizer': 'Adam', 'lr': .01, 'batch_size': 16, 'epochs': 2}
preview_net, _ = breast_cancer(x_train, y_train, x_val, y_val, preview_params)
assert len(preview_net.history['val_loss']) == 2
```

The current `(history, model)` return is supported alongside the historical `(net, net.parameters())` return shown here:

```python
legacy_result = (preview_net, preview_net.parameters())
normalized = talos.backends.normalise_result(legacy_result, backend='torch')
print(normalized['metrics'])  # Final values from the real training history.
assert normalized['model'] is preview_net

# The current return passes the history dictionary and trained network directly.
modern_result = (preview_net.history, preview_net)
print(talos.backends.normalise_result(modern_result, backend='torch')['metrics'])
```

### Parameter dictionary

```python
p = {'first_neuron': [8, 16], 'second_neuron': [4], 'dropout': [.1],
     'optimizer': ['Adam'], 'lr': [.01], 'batch_size': [16], 'epochs': [2]}
```

The parameter dictionary accepts candidate lists or range tuples in the form `(min, max, number_of_values)`.

### Scan()

```python
scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         params=p, model=breast_cancer, experiment_name='breast_cancer_torch',
                         round_limit=2, seed=17, backend='torch')
assert len(scan_object.data) == 2
predicted = talos.Predict(scan_object).predict_classes(
    x_val, metric='val_loss', asc=True, task='multi_class')
assert predicted.shape == (len(x_val),)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

The module-level `build_network` factory and `talos_config` describe reconstruction for native Torch archives. The [complete example](Examples_PyTorch_Code.md) guards training in `run_example()` so importing its factory during restore cannot retrain. Save it as an importable Python module for portable restore; a factory defined only in `__main__` must be supplied explicitly to `Restore(model_factory=build_network)`. Validation runs in evaluation mode under `torch.no_grad()`.

## Expected result

`scan_object.data` contains two completed rows; the final block produces a class index for each validation row. `TorchHistory` holds two validation-loss observations per trial. Its metric helper uses the names `metric` and `val_metric`; in this recipe those values are the explicitly computed F1 scores, not classification accuracy. The run directory contains `results.csv` and checkpoint artifacts; inspect `scan_object.run_dir` for its location.

## Failure boundaries

This callback owns the optimizer, gradient updates and train/evaluation modes. Keep integer class labels and two output logits consistent with cross-entropy. Histories must contain numeric scalar metrics; use finite values for meaningful candidate ranking. A factory defined only in `__main__` needs an explicit `Restore(model_factory=build_network)`; use the guarded importable complete program for portable source-based restoration.

If an import fails, check the active interpreter and [installation options](Install_Options.md). If a scan fails before its first trial, compare the data shapes, parameter keys and callback return with the [Scan contract](Scan.md).

## Read next

[Backends](Backends.md) describes both supported callback returns. [Deploy](Deploy.md) and [Restore](Restore.md) describe the native Torch archive and factory contract.
