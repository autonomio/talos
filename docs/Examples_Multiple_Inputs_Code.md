[BACK](Examples_Multiple_Inputs.md)

# Multiple Inputs

```python
import talos
import wrangle

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.utils import plot_model
from keras.layers.merge import concatenate

x, y = talos.templates.datasets.iris()
x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .5)

def iris_multi(x_train, y_train, x_val, y_val, params):

    # the first side of the network
    first_input = Input(shape=(4,))
    first_hidden1 = Dense(params['left_neurons'], activation=params['activation'])(first_input)
    first_hidden2 = Dense(params['left_neurons'], activation=params['activation'])(first_hidden1)

    # the second side of the network
    second_input = Input(shape=(4,))
    second_hidden1 = Dense(params['right_neurons'], activation=params['activation'])(second_input)
    second_hidden2 = Dense(params['right_neurons'], activation=params['activation'])(second_hidden1)
    third_hidden2 = Dense(params['right_neurons'], activation=params['activation'])(second_hidden2)

    # merging the two networks
    merged = concatenate([first_hidden2, first_hidden2])

    # creating the output
    output = Dense(3, activation='softmax')(merged)

    # put the model together, compile and fit
    model = Model(inputs=[first_input, second_input], outputs=output)
    model.compile('adam',
                  'binary_crossentropy',
                  metrics=['acc', talos.utils.metrics.f1score])

    out = model.fit(x=[x_train, x_train],
                    y=y_train,
                    validation_data=[[x_val, x_val], y_val],
                    epochs=150,
                    batch_size=params['batch_size'],
                    verbose=0)

    return out, model


p = {'activation':['relu', 'elu'],
     'left_neurons': [10, 20, 30],
     'right_neurons': [10, 20, 30],
     'batch_size': [15, 20, 25]}

scan_object = talos.Scan(x=x_train,
                         y=y_train,
                         x_val=x_val,
                         y_val=y_val,
                         params=p,
                         model=iris_multi)
```
