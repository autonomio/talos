[BACK](Examples_Multiple_Outputs.md)

# Multiple Outputs

```python
import talos
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
x, diagnosis = load_breast_cancer(return_X_y=True)
# Predict diagnosis and the separately measured mean radius, without input leakage.
radius = x[:, 0:1].astype('float32') / 30
features = x[:, 1:].astype('float32')
x_train, x_val, diagnosis_train, diagnosis_val, radius_train, radius_val = train_test_split(
    features, diagnosis, radius, train_size=144, test_size=36,
    stratify=diagnosis, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val = scaler.transform(x_train), scaler.transform(x_val)
y_train = [diagnosis_train, radius_train]
y_val = [diagnosis_val, radius_val]

def breast_cancer_multi(x_train, y_train, x_val, y_val, params):
    input_layer = Input(shape=(x_train.shape[1],))
    hidden = Dense(params['neurons'], activation=params['activation'])(input_layer)
    diagnosis = Dense(1, activation='sigmoid', name='diagnosis')(hidden)
    radius = Dense(1, name='radius')(hidden)
    model = Model(inputs=input_layer, outputs=[diagnosis, radius])
    model.compile(optimizer='adam',
                  loss={'diagnosis': 'binary_crossentropy', 'radius': 'mse'},
                  metrics={'diagnosis': ['accuracy', talos.utils.metrics.f1score],
                           'radius': ['mae']})
    out = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return out, model

p = {'activation': ['relu', 'elu'], 'neurons': [8],
     'batch_size': [16], 'epochs': [2]}

scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         params=p, model=breast_cancer_multi,
                         experiment_name='breast_cancer_multi_output', round_limit=2,
                         seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```
