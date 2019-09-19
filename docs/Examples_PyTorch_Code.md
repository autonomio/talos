[BACK](Examples_PyTorch.md)

# PyTorch Example

```python
import talos
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_optimizer import torch_optimizer

from sklearn.metrics import f1_score

# load the data
x, y = talos.templates.datasets.breast_cancer()
x = talos.utils.rescale_meanzero(x)
x_train, y_train, x_val, y_val = talos.utils.val_split(x, y, .2)

# convert arrays to tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).long()
x_val = torch.from_numpy(x_val).float()
y_val = torch.from_numpy(y_val).long()

def breast_cancer(x_train, y_train, x_val, y_val, params):

    # takes in a module and applies the specified weight initialization
    def weights_init_uniform_rule(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)

    class BreastCancerNet(nn.Module, talos.utils.TorchHistory):

        def __init__(self, n_feature):

            super(BreastCancerNet, self).__init__()
            self.hidden = torch.nn.Linear(n_feature, params['first_neuron'])
            torch.nn.init.normal_(self.hidden.weight)
            self.hidden1 = torch.nn.Linear(params['first_neuron'], params['second_neuron'])
            self.dropout = torch.nn.Dropout(params['dropout'])
            self.out = torch.nn.Linear(params['second_neuron'], 2)

        def forward(self, x):

            x = F.relu(self.hidden(x))
            x = self.dropout(x)
            x = torch.sigmoid(self.hidden1(x))
            x = self.out(x)
            return x


    net = BreastCancerNet(x_train.shape[1])
    net.apply(weights_init_uniform_rule)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch_optimizer(net,
                                params['optimizer'],
                                lr=params['lr'],
                                momentum=params['momentum'],
                                weight_decay=params['weight_decay'])

    # Initialize history of net
    net.init_history()

    for epoch in range(params['epochs']):

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = net(x_train)

        # calculate accuracy
        prediction = torch.max(outputs, 1)[1]
        metric = f1_score(y_train.data, prediction.data)

        # calculate loss + backward + optimize
        loss = loss_func(outputs, y_train)
        loss.backward()
        optimizer.step()

        # calculate accuracy for validation data
        output_val = net(x_val)
        prediction = torch.max(output_val, 1)[1]
        val_metric = f1_score(y_val.data, prediction.data)

        # calculate loss for validation data
        val_loss = loss_func(output_val, y_val)

        # append history
        net.append_loss(loss.item())
        net.append_metric(metric)
        net.append_val_loss(val_loss.item())
        net.append_val_metric(val_metric)


    # Get history object
    return net, net.parameters()


p = {'activation':['relu', 'elu'],
       'optimizer': ['Nadam', 'Adam'],
       'losses': ['logcosh'],
       'hidden_layers':[0, 1, 2],
       'batch_size': (20, 50, 5),
       'epochs': [10, 20]}

scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=breast_cancer,
                         experiment_name='breast_cancer',
                         round_limit=100)
```
