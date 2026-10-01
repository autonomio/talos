# Energy Draw Callback

A callback for recording sampled GPU power draw (watts) on epoch begin and end. Physical measurements require supported NVIDIA hardware and `nvidia-smi`; endpoint averaging estimates watt-seconds rather than measuring energy continuously. The callback allows:

- record and analyze model energy consumption
- optimize towards energy efficient models

### how-to-use

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
