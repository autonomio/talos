## LR Normalizer

As one experiment may include more than one optimizer, and optimizers generally have default learning rates in different order of magnitudes, lr_normalizer can be used to allow simultanously including different optimizers and different degrees of learning rates into the Talos experiment.

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

<aside class="notice">
The lr_normalizer needs to be invoked explicitly
</aside>
