# Typical Keras sweep: complete code

Train two small Iris classifiers with different activations. This page is the standalone companion to the [walkthrough](Examples_Typical.md).

## Prerequisites and execution

Use Python 3.11–3.13 with the [TensorFlow extra](Backends.md) (`talos[tensorflow]`). The dataset is an offline scikit-learn fixture. From a writable experiment directory, save the following program as `iris_example.py` and execute `python iris_example.py`.

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

def iris_model(x_train, y_train, x_val, y_val, params):
    model = Sequential([Input(shape=(4,)),
                        Dense(params['first_neuron'], activation=params['activation']),
                        Dense(3, activation='softmax')])
    model.compile(optimizer=params['optimizer'], loss=params['losses'],
                  metrics=['accuracy', talos.utils.metrics.f1score])
    out = model.fit(x_train, y_train, batch_size=params['batch_size'],
                    epochs=params['epochs'], validation_data=(x_val, y_val),
                    verbose=0)
    return out, model

p = {'activation': ['relu', 'elu'],
     'first_neuron': [8], 'optimizer': ['adam'],
     'losses': ['categorical_crossentropy'],
     'batch_size': [16], 'epochs': [2]}

scan_object = talos.Scan(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                         model=iris_model, params=p, experiment_name='iris',
                         round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```

`Scan()` always needs to have `x`, `y`, `model`, and `params` arguments declared. Find the description for all `Scan()` arguments [Scan arguments](Scan.md#arguments).

## Result and failure boundaries

`scan_object.data` contains two completed rows. Each row records the selected activation and final training/validation metrics. The model returns a probability vector for each of the three Iris classes. The program writes result and checkpoint artifacts to its experiment run directory.

The Iris labels are one-hot vectors, so the output has three units and uses categorical cross-entropy. Keep this encoding, the output shape and the loss consistent when replacing the dataset.

## Read next

Return to the [walkthrough](Examples_Typical.md) for the procedure and failure diagnosis. [Analyze results](Analyze.md), then [evaluate candidates](Evaluate.md) on data held out from tuning.
