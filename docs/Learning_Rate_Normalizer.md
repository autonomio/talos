# Learning-rate normalizer

`talos.model.lr_normalizer` converts a shared learning-rate scale to optimizer-specific values using fixed divisors. Import it from `talos.model.normalizers`, `talos.model`, or `talos.utils`. Invoke it explicitly when constructing an optimizer; adding an `lr` candidate to Scan does not apply normalization automatically.

The normalizer requires only the Talos core. The runnable fit example below additionally requires the [TensorFlow extra](Install_Options.md) and scikit-learn, included with the core.

```python
from talos.model.normalizers import lr_normalizer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris
x, y = load_iris(return_X_y=True)
x, y = x[y < 2].astype('float32'), y[y < 2]
params = {'optimizer': Adam, 'lr': .5}
model = Sequential([Input(shape=(4,)), Dense(1, activation='sigmoid')])
model.compile(loss='binary_crossentropy',
              optimizer=params['optimizer'](learning_rate=lr_normalizer(params['lr'], params['optimizer'])),
              metrics=['accuracy'])
history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
assert len(history.history['loss']) == 1
```

The example fits one epoch with an Adam learning rate of `0.0005`, obtained from the normalized input `0.5`. Its returned History contains one loss value.

## Interface and scale

The signature is `lr_normalizer(lr, optimizer)`. `lr` is a numeric scale and `optimizer` is an optimizer class or instance. The helper uses its class name and returns `lr / divisor`; it does not instantiate or mutate the optimizer.

| Optimizer name | Divisor | Result for `lr=1` |
| --- | --- | --- |
| `SGD` | 100 | `0.01` |
| `Adagrad` | 100 | `0.01` |
| `Adam` | 1000 | `0.001` |
| `RMSprop` | 1000 | `0.001` |
| `Adamax` | 500 | `0.002` |

These are Talos' fixed scaling conventions, independent of a framework's current default optimizer settings. The normalized input is not itself the optimizer's learning rate. Supported names may come from Keras, TensorFlow or Torch, but each framework's optimizer construction remains caller-owned.

## Failure boundaries

An unsupported optimizer class name raises `TalosModelError`. The helper performs no positivity, schedule or framework-compatibility validation on `lr`; numeric division or the downstream optimizer may reject unsuitable input. A custom class with a supported name receives that name's divisor.

## Read next

Use [AutoParams](AutoParams.md) to generate learning-rate and optimizer candidates, or [Scan](Scan.md#input-model) to apply the value in a custom callback.
