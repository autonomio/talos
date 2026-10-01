# Metrics

Several common and useful performance metrics are available through `talos.utils.metrics`:

- matthews
- precision
- recall
- fbeta
- f1score
- mae
- mse
- rmae
- rmse
- mape
- msle
- rmsle

You can use these metrics as you would Keras metrics:

```python
import talos
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
x, y = talos.templates.datasets.iris()
model = Sequential([Input(shape=(4,)), Dense(3, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', talos.utils.metrics.f1score])
history = model.fit(x, y, epochs=1, batch_size=16, verbose=0)
assert len(history.history['f1score']) == 1
```
If you would like to add new metrics to Talos, make a [feature request](https://github.com/autonomio/talos/issues/new) or create a [pull request](https://github.com/autonomio/talos/compare).

Classification metric objects accumulate counts across batches. `fbeta` is a stateless batch score; use `classification_metric(beta=...)` for an epoch-wide score. Continuous root helpers reduce over each example’s target axis before Keras averages samples.
