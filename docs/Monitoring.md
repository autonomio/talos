# Monitoring

There are several options for monitoring the experiment.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
# turn off progress bar
quiet_scan = talos.Scan(x, y, p, input_model, 'quiet', x_val=x_val, y_val=y_val,
                       disable_progress_bar=True, seed=17)

# enable live training plot
from talos.callbacks import TrainingPlot

_, model = input_model(x, y, x_val, y_val, {key: values[0] for key, values in p.items()})
out = model.fit(x,
                y,
                epochs=2,
                callbacks=[TrainingPlot()])

# turn on parameter printing
printed_scan = talos.Scan(x, y, p, input_model, 'printed', x_val=x_val, y_val=y_val,
                         print_params=True, disable_progress_bar=True, seed=17)
```

**Progress Bar :** A round-by-round updating progress bar that shows the remaining rounds, together with a time estimate to completion. Progress bar is on by default.

**Live Monitoring :** Live monitoring provides an epoch-by-epoch updating line graph that is enabled through the `TrainingPlot()` custom callback.

**Round Hyperparameters :** Displays the hyperparameters for each permutation. It may be combined with callback plots; terminal output and plotting are separate views.

### Local Monitoring

Epoch-by-epoch training data is available during the experiment using the `ExperimentLog`:

```python
params = {key: values[0] for key, values in p.items()}
_, model = input_model(x, y, x_val, y_val, params)
history = model.fit(x, y, validation_data=(x_val, y_val), epochs=2, verbose=0,
                    callbacks=[talos.callbacks.ExperimentLog('epoch_logs', params)])
```
Here `params` is the parameter dictionary for one training trial. Inside Scan, `ExperimentLog` uses the active trial/run context automatically; outside Scan, the name gives the local output folder. Each callback writes an epoch log with a trial identifier.
