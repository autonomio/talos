# Generator

`talos.utils.generator` yields repeating NumPy batches; `talos.utils.SequenceGenerator` provides finite, indexed batches for Keras `model.fit()`. Import them from `talos.utils`. They batch caller-supplied arrays without acquiring data, splitting validation sets or changing a parameter sweep. Custom framework generators can be used in a training callback as usual.

The basic generator uses the Talos core only. SequenceGenerator imports the selected [Keras or TensorFlow backend](Backends.md). The Iris acquisition helper in the example reads a remote Autonomio CSV; see [templates](Templates.md) for its dataset boundary.

## Repeating batches

```python
import talos
x, y = talos.templates.datasets.iris()
stream = talos.utils.generator(x=x, y=y, batch_size=20)
x_batch, y_batch = next(stream)
assert x_batch.shape == (20, 4) and y_batch.shape == (20, 3)
```

## Indexed batches

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

Framework scheduling and prefetch behavior affect training throughput. Choose batch sizes and data pipelines against the actual workload; these conveniences do not promise a performance improvement.

For historical context, read the Talos thread on [using fit_generator](https://github.com/autonomio/talos/issues/11).

## Interfaces and results

| Surface | Signature and result |
| --- | --- |
| `generator` | `generator(x, y, batch_size)` returns an infinite iterator yielding `(x_batch, y_batch)` arrays cast to `float32`. |
| `SequenceGenerator` | `SequenceGenerator(x_set=None, y_set=None, batch_size=32, *, x=None, y=None, backend='keras')` returns a framework Sequence instance. `x_set`/`y_set` take precedence over the keyword aliases. |

Both preserve source row order and include the final partial batch. The repeating generator starts again after the final batch; the caller must bound iteration or supply the framework's required step count. SequenceGenerator's length is `ceil(len(x) / batch_size)`, indexing slices the original inputs, and its output retains their dtypes. Neither helper shuffles batches between epochs.

The first example yields shapes `(20, 4)` and `(20, 3)`. With 150 rows and a batch size of 20, the indexed sequence has eight batches, with ten rows in the final batch. Its fit example completes one epoch and returns one recorded loss value.

## Input boundaries

Provide nonempty, sliceable feature and label arrays with matching lengths and a positive integer batch size. The basic generator does not validate those conditions; invalid input may yield empty batches, division errors or slicing errors. SequenceGenerator raises `ValueError` for missing x/y data or `batch_size < 1`, but does not validate equal sample counts or every index. List-based multiple inputs and targets require a caller-owned generator that preserves their structure.

A missing framework fails only when SequenceGenerator is constructed. For native Torch, use a Torch data pipeline inside the model callback rather than passing a Keras Sequence to a Torch training loop.

## Read next

Use the [generator walkthrough](Examples_Generator.md) to integrate batches into Scan, or [Scan](Scan.md#input-model) to supply your own training callback.
