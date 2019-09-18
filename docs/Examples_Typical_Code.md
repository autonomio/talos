[BACK](Examples_Typical.md)

# Typical Case Example

```python
import talos as ta
from keras.models import Sequential
from keras.layers import Dense

x, y = ta.templates.datasets.iris()

def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    model.add(Dense(32, input_dim=4, activation=params['activation']))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer=params['optimizer'], loss=params['losses'])

    out = model.fit(x_train, y_train,
                     batch_size=params['batch_size'],
                     epochs=params['epochs'],
                     validation_data=[x_val, y_val],
                     verbose=0)

    return out, model

p = {'activation':['relu', 'elu'],
       'optimizer': ['Nadam', 'Adam'],
       'losses': ['logcosh'],
       'hidden_layers':[0, 1, 2],
       'batch_size': (20, 50, 5),
       'epochs': [10, 20]}

scan_object = ta.Scan(x, y, model=iris_model, params=p, fraction_limit=0.1)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
