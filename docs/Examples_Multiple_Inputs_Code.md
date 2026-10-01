# Multiple-input Keras sweep: complete code

Train a Keras model with two aligned feature arrays. This page is the standalone companion to the [walkthrough](Examples_Multiple_Inputs.md).

## Prerequisites and execution

Use Python 3.11–3.13 with the [TensorFlow extra](Backends.md) (`talos[tensorflow]`). The dataset is an offline scikit-learn fixture. From a writable experiment directory, save the following program as `multiple_inputs.py` and execute `python multiple_inputs.py`.

The walkthrough owns the data split, callback explanation and interpretation of metrics. The program is bounded to two trials; successful execution passes its result-row assertion.

## Program

```python
import talos
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten, concatenate

x, y = talos.templates.datasets.iris()
x_train, x_val, y_train, y_val = train_test_split(
    x.astype('float32'), y.astype('float32'), test_size=.2, random_state=17,
    stratify=y.argmax(axis=1))

def iris_multi(x_train, y_train, x_val, y_val, params):
    # Each input contains a different pair of measured Iris features.
    first_input = Input(shape=(2,))
    first_hidden = Dense(params['left_neurons'], activation=params['activation'])(first_input)
    second_input = Input(shape=(2,))
    second_hidden = Dense(params['right_neurons'], activation=params['activation'])(second_input)
    merged = concatenate([first_hidden, second_hidden])
    output = Dense(3, activation='softmax')(merged)
    model = Model(inputs=[first_input, second_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', talos.utils.metrics.f1score])
    out = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'], verbose=0)
    return out, model

p = {'activation': ['relu', 'elu'], 'left_neurons': [8],
     'right_neurons': [8], 'batch_size': [16], 'epochs': [2]}

scan_object = talos.Scan(x=[x_train[:, :2], x_train[:, 2:]], y=y_train,
                         x_val=[x_val[:, :2], x_val[:, 2:]], y_val=y_val,
                         params=p, model=iris_multi, multi_input=True,
                         experiment_name='iris_multi_input', round_limit=2,
                         seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

## Result and failure boundaries

`scan_object.data` contains two completed rows. The final prediction block returns one three-class probability vector per validation row. The two input arrays preserve the order of the corresponding target rows. The program writes result and checkpoint artifacts to its experiment run directory.

Keep both arrays aligned and pass them in the same order to training, validation and prediction. Each input layer expects two columns; a single four-column array does not match this model. Keep `multi_input=True` when using the established multi-input facade.

## Read next

Return to the [walkthrough](Examples_Multiple_Inputs.md) for the procedure and failure diagnosis. [Predict](Predict.md) describes candidate selection and input forwarding; [multiple outputs](Examples_Multiple_Outputs.md) covers aligned target lists.
