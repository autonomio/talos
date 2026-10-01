# Generator

Talos provides two data generators to be used with `model.fit()`. You can of course use your own generator as you would use it otherwise with stand-alone Keras.

#### basic data generator

```python
import talos
x, y = talos.templates.datasets.iris()
stream = talos.utils.generator(x=x, y=y, batch_size=20)
x_batch, y_batch = next(stream)
assert x_batch.shape == (20, 4) and y_batch.shape == (20, 3)
```

#### sequence generator
```python
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
batches = talos.utils.SequenceGenerator(x=x, y=y, batch_size=20, backend='tensorflow')
assert sum(len(batches[i][0]) for i in range(len(batches))) == len(x)
model = Sequential([Input(shape=(4,)), Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy')
history = model.fit(batches, epochs=1, verbose=0)
assert len(history.history['loss']) == 1
```


NOTE: There are many performance considerations that come with using data generators in Keras. If you run in to performance issues, learn more about the experiences of other Keras users online.

For historical context, read the Talos thread on [using fit_generator](https://github.com/autonomio/talos/issues/11).
