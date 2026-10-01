# Keras sequence-generator sweep: complete code

Train two digit classifiers using a Talos sequence generator. This page is the standalone companion to the [walkthrough](Examples_Generator.md).

## Prerequisites and execution

Use Python 3.11–3.13 with the [TensorFlow extra](Backends.md) (`talos[tensorflow]`). The dataset is an offline scikit-learn fixture. From a writable experiment directory, save the following program as `generator_example.py` and execute `python generator_example.py`.

The walkthrough owns the data split, callback explanation and interpretation of metrics. The program is bounded to two trials; successful execution passes its result-row assertion.

## Program

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

## Result and failure boundaries

`scan_object.data` contains two completed rows. Each callback trains from `SequenceGenerator` batches and validates on the separate 36-row array split. The network produces ten class probabilities per image. The program writes result and checkpoint artifacts to its experiment run directory.

The arrays must match `(8, 8, 1)` image inputs and integer digit labels. Use sparse categorical cross-entropy for those labels. Modern Keras does not accept `workers` in `fit()`; configure a supported sequence/PyDataset instead. See [Generator](Generator.md) for replayability limits when using external streams.

## Read next

Return to the [walkthrough](Examples_Generator.md) for the procedure and failure diagnosis. [Generator](Generator.md) covers the input contract; [Analyze](Analyze.md) covers the completed sweep.
