"""PyTorch template with an importable reconstruction factory and native weights."""
backend = 'torch'


def params():
    return {'neurons': [8, 16], 'learning_rate': [.01, .03], 'epochs': [3], 'batch_size': [16]}


def prep(data, round_params):
    # The caller supplies data. A CLI entry point can use a user-owned loader here.
    if data is None:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        x, y = load_iris(return_X_y=True)
        x_train, x_val, y_train, y_val = train_test_split(x, y, stratify=y, test_size=.2, random_state=17)
        return {'x_train': x_train, 'y_train': y_train, 'x_val': x_val, 'y_val': y_val}
    return data


def make_model(neurons=8):
    from torch import nn
    network = nn.Sequential(nn.Linear(4, neurons), nn.ReLU(), nn.Linear(neurons, 3))
    network.talos_factory = make_model
    network.talos_config = {'neurons': neurons}
    return network


def model(data, round_params):
    import torch

    from talos.utils import TorchHistory
    network = make_model(round_params['neurons'])
    optimizer = torch.optim.Adam(network.parameters(), lr=round_params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    x_train, y_train = torch.as_tensor(data['x_train'], dtype=torch.float32), torch.as_tensor(data['y_train'], dtype=torch.long)
    x_val, y_val = torch.as_tensor(data['x_val'], dtype=torch.float32), torch.as_tensor(data['y_val'], dtype=torch.long)
    history = TorchHistory()
    for epoch in range(round_params['epochs']):
        network.train()
        for start in range(0, len(x_train), round_params['batch_size']):
            optimizer.zero_grad()
            loss = criterion(network(x_train[start:start + round_params['batch_size']]), y_train[start:start + round_params['batch_size']])
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            history.append_loss(criterion(network(x_train), y_train).item())
            history.append_val_loss(criterion(network(x_val), y_val).item())
    return {'_model': network, '_history': history, 'backend': 'torch', 'model_factory': make_model}
