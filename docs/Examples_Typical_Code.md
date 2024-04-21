[BACK](Examples_Typical.md)

# Typical Case Example

```python
import talos as talos
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x, y = talos.templates.datasets.iris()

# define the model
def iris_model(x_train, y_train, x_val, y_val, params):

    model = Sequential()
    
    model.add(Dense(32, input_dim=4, activation=params['activation']))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(optimizer=params['optimizer'],
                  loss=params['losses'],
                  metrics=[talos.utils.metrics.f1score])

    out = model.fit(x_train, y_train,
                     batch_size=params['batch_size'],
                     epochs=params['epochs'],
                     validation_data=[x_val, y_val],
                     verbose=0)

    return out, model

# set the parameter space boundaries
p = {'activation':['relu', 'elu'],
     'optimizer': ['Adagrad', 'Adam'],
     'losses': ['categorical_crossentropy'],
     'epochs': [100, 200],
     'batch_size': [4, 6, 8]}

# start the experiment
scan_object = talos.Scan(x=x, 
                         y=y, 
                         model=iris_model,
                         params=p,
                         experiment_name='iris',
                         round_limit=20)
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
