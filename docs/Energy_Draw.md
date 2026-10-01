# Energy draw callback

`talos.callbacks.PowerDraw` records sampled GPU power draw in watts at epoch begin and end. Import it from `talos.callbacks`; `talos.utils.power_draw_append` adds the summary to a training History object.

Install the matching [Keras or TensorFlow backend](Backends.md). Physical measurements additionally require supported NVIDIA hardware and `nvidia-smi`; an explicit `provider` makes callback behavior testable on a CPU. Endpoint averaging estimates watt-seconds rather than measuring energy continuously. The callback allows:

- record and analyze model energy consumption
- optimize towards energy efficient models

## Example

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

Before `model.fit()` in the input model:

```python
from talos.callbacks import PowerDraw
# A provider fixture makes this CPU example runnable; these are declared watts.
power_draw = PowerDraw(provider=lambda: 10.0)
```

Then use `power_draw` as you would callbacks in general:

```python
params = {key: values[0] for key, values in p.items()}
_, model = input_model(x, y, x_val, y_val, params)
history = model.fit(x, y, validation_data=(x_val, y_val), epochs=2, verbose=0,
                    callbacks=[power_draw])
```

To get the energy draw data into the experiment log:

```python
history = talos.utils.power_draw_append(history, power_draw)
assert history.history['watts_min'] == [10.0]
assert history.history['Ws'][0] >= 0
```

NOTE: this line has to be after `model.fit()`.

For physical sampling omit `provider` and select the GPU with `PowerDraw(device=0)`. The fixture above verifies callback/log behavior only; it is not a measured hardware result.

## Interface and units

The callback signature is `PowerDraw(device=0, backend='keras', provider=None)`. `device` sets the `-i` device selector for `nvidia-smi`; `backend` selects the callback base. A `provider` is a zero-argument callable returning a numeric power reading in watts. Without a provider, each endpoint launches an `nvidia-smi` subprocess for that device.

`.log` contains `epoch_begin` and `epoch_end` lists in watts and a `seconds` list of monotonic elapsed epoch durations. Calling `power_draw_append(history, power_draw)` modifies `history.history` in place and returns that same History object. It adds one-element lists for the entire fit:

| Field | Unit and calculation |
| --- | --- |
| `watts_min` | Watts; minimum of all endpoint readings. |
| `watts_max` | Watts; maximum of all endpoint readings. |
| `seconds` | Seconds; sum of recorded epoch durations. |
| `Ws` | Watt-seconds, equivalent to joules; sum of each epoch's mean endpoint power times its duration, rounded to two decimal places. |

This estimate covers the selected GPU's epoch intervals. It does not include total machine power, dataset preparation, idle intervals or continuous power integration. The example's fixed provider proves the expected fields and nonnegative duration; it is not evidence of hardware energy use.

## Failure boundaries

Append only after at least one completed epoch, with paired begin/end readings; an empty log has no minimum or maximum. Missing `nvidia-smi`, unavailable devices, subprocess failures and nonnumeric provider output propagate as errors. Native Torch loops must invoke the callback hooks explicitly with `backend='torch'`; Talos does not attach Keras callbacks to a Torch training loop automatically.

## Read next

See [monitoring](Monitoring.md) for epoch logs and plots, and [Scan](Scan.md) to include these history summaries in a parameter sweep.
