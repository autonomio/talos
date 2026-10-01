[BACK](Examples_Generator.md)

# Generator

```python
import talos
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate
from talos.utils import SequenceGenerator

from sklearn.datasets import load_digits
x, y = load_digits(return_X_y=True)
x = x.reshape(-1, 8, 8, 1).astype('float32') / 16
x_train, x_val, y_train, y_val = train_test_split(
    x, y, train_size=144, test_size=36, stratify=y, random_state=17)

def digits_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(8, 8, 1)),
                        Conv2D(4, (3, 3), activation=params['activation']),
                        Flatten(), Dense(8, activation=params['activation']),
                        Dropout(params['dropout']), Dense(10, activation='softmax')])
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=['accuracy', talos.utils.metrics.f1score])
    batches = SequenceGenerator(x=x_train, y=y_train,
                                batch_size=params['batch_size'], backend='tensorflow')
    out = model.fit(batches, epochs=params['epochs'],
                    validation_data=(x_val, y_val), verbose=0)
    return out, model

p = {'activation': ['relu', 'elu'], 'optimizer': ['adam'],
     'losses': ['sparse_categorical_crossentropy'], 'dropout': [.1],
     'batch_size': [16], 'epochs': [2]}

scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         model=digits_model, params=p, experiment_name='digits_generator',
                         round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```
