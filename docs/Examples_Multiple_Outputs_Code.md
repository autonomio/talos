[BACK](Examples_Multiple_Outputs.md)

# Multiple Outputs

```python
import talos
import wrangle

from keras.layers import Input, Dense, Dropout
from keras.models import Model

x, y = talos.templates.datasets.telco_churn()

x_train, y1_train, x_val, y1_val = wrangle.array_split(x, y[0], 0.3)
x_train, y2_train, x_val, y2_val = wrangle.array_split(x, y[1], 0.3)

def telco_churn(x_train, y_train, x_val, y_val, params):

    # the second side of the network
    input_layer = Input(shape=(42,))
    hidden_layer1 = Dense(params['neurons'], activation=params['activation'])(input_layer)
    hidden_layer2 = Dense(params['neurons'], activation=params['activation'])(hidden_layer1)
    hidden_layer3 = Dense(params['neurons'], activation=params['activation'])(hidden_layer2)

    # creating the outputs
    output1 = Dense(1, activation='sigmoid', name='loss_function')(hidden_layer3)
    output2 = Dense(1,  activation='sigmoid', name='f1_metric')(hidden_layer3)

    losses = {"loss_function": "binary_crossentropy",
              "f1_metric": "binary_crossentropy"}

    loss_weights = {"loss_function": 1.0, "f1_metric": 1.0}

    # put the model together, compile and fit
    model = Model(inputs=input_layer, outputs=[output1, output2])

    model.compile('adam', loss=losses, loss_weights=loss_weights,
                  metrics=['acc', talos.utils.metrics.f1score])

    out = model.fit(x=x_train,
                    y=y_train,
                    validation_data=[x_val, y_val],
                    epochs=150,
                    batch_size=params['batch_size'],
                    verbose=0)

    return out, model

p = {'activation':['relu', 'elu'],
     'neurons': [10, 20, 30],
     'batch_size': [15, 20, 25]}

scan_object = talos.Scan(x=x_train,
                         y=[y1_train, y2_train],
                         x_val=x_val,
                         y_val=[y1_val, y2_val],
                         params=p,
                         model=telco_churn)
```
