"""Caller-owned PyTorch training; supply tensors/arrays and optional task config."""
backend = 'torch'



def params():
    return {'units': [16, 32], 'epochs': [5], 'learning_rate': [0.001]}


def prep(context, round_params=None):
    if context is None:
        raise ValueError('Supply caller data to execute(), or implement data loading in this prep')
    return context


def build_model(input_size, units, outputs):
    import torch
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(input_size, units),
                               torch.nn.ReLU(), torch.nn.Linear(units, outputs))


def model(prepared, round_params):
    import torch
    import numpy as np
    x = torch.as_tensor(np.asarray(prepared['x_train']), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(prepared['y_train']))
    task = prepared.get('task', 'regression')
    if task == 'multiclass':
        outputs = y.shape[-1] if y.ndim > 1 else int(y.max().item()) + 1
        target = y.argmax(dim=-1).long() if y.ndim > 1 else y.long().reshape(-1)
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        outputs = y.shape[-1] if y.ndim > 1 else 1
        target = y.float().reshape(len(y), outputs)
        loss_fn = torch.nn.BCEWithLogitsLoss() if task == 'binary' else torch.nn.MSELoss()
    configuration = {'input_size': x[0].numel(), 'units': round_params['units'], 'outputs': outputs}
    network = build_model(**configuration)
    network.talos_factory = build_model
    network.talos_config = configuration
    optimizer = torch.optim.Adam(network.parameters(), lr=round_params['learning_rate'])
    history = {'loss': []}
    for _ in range(round_params['epochs']):
        network.train()
        optimizer.zero_grad()
        loss = loss_fn(network(x), target)
        loss.backward()
        optimizer.step()
        history['loss'].append(float(loss.detach()))
    network.eval()
    return history, network
