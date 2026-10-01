[BACK](Examples_PyTorch.md)

# PyTorch Example

```python
import talos
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


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


def run_example():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=144, test_size=36, stratify=y, random_state=17)
    scaler = StandardScaler().fit(x_train)
    x_train = torch.as_tensor(scaler.transform(x_train), dtype=torch.float32)
    x_val = torch.as_tensor(scaler.transform(x_val), dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long)
    y_val = torch.as_tensor(y_val, dtype=torch.long)
    p = {'first_neuron': [8, 16], 'second_neuron': [4], 'dropout': [.1],
         'optimizer': ['Adam'], 'lr': [.01], 'batch_size': [16], 'epochs': [2]}
    scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                             params=p, model=breast_cancer, experiment_name='breast_cancer_torch',
                             round_limit=2, seed=17, backend='torch')
    assert len(scan_object.data) == 2
    predicted = talos.Predict(scan_object).predict_classes(
        x_val, metric='val_loss', asc=True, task='multi_class')
    assert predicted.shape == (len(x_val),)
    return scan_object


if __name__ == "__main__":
    run_example()
```

Save this code as an importable Python module. Importing it defines the model and factory without starting training. Call `run_example()` to train; the `__main__` guard also trains when you execute the file directly. The importable `build_network` factory makes archives portable. When a factory is defined only in `__main__`, supply it explicitly to `Restore(model_factory=build_network)`.
