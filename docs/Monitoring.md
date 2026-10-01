# Monitoring

Talos exposes round progress, printed parameters, per-epoch CSV logs and Keras training plots. Scan owns round-level progress; callbacks observe the caller's training loop. This page covers `talos.callbacks.TrainingPlot` and `ExperimentLog`, plus Scan's output flags.

Prerequisites are the shared [Scan setup](Scan.md#minimal-example) and its installed [framework backend](Backends.md). TrainingPlot additionally requires the `plots` extra. In a headless process, select a Matplotlib backend suitable for that environment before plotting.

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

## Local epoch logs

Epoch-by-epoch training data is available during the experiment using the `ExperimentLog`:

```python
params = {key: values[0] for key, values in p.items()}
_, model = input_model(x, y, x_val, y_val, params)
history = model.fit(x, y, validation_data=(x_val, y_val), epochs=2, verbose=0,
                    callbacks=[talos.callbacks.ExperimentLog('epoch_logs', params)])
```

Here `params` is the parameter dictionary for one training trial. Inside Scan, `ExperimentLog` uses the active trial/run context automatically; outside Scan, the name gives the local output folder. Each callback writes an epoch log with a trial identifier.

## Interfaces and side effects

| Surface | Defaults and result |
| --- | --- |
| Scan progress | `disable_progress_bar=False` enables a round progress bar. The display reports remaining rounds and a time estimate; it does not prove runtime duration in advance. |
| Scan parameters | `print_params=False`; set `True` to print each trial's parameter values. |
| `TrainingPlot(backend='keras', **kwargs)` | Returns a framework callback. Current extra keyword arguments are accepted but unused. Creates a Matplotlib figure at training start and redraws all reported metric series after each epoch. |
| `ExperimentLog(experiment_name, params, backend='keras')` | Returns a framework callback and creates the output folder when constructed. Writes an `epochs-<trial-id>.log` CSV during fitting. |

TrainingPlot exposes `.history`, `.figure` and `.axes` after training begins. Plot lines include the metric keys supplied by the training callback logs. ExperimentLog writes trial ID, one-based epoch number, metric values and a JSON representation of the trial parameters. Inside a Talos trial it uses the active run directory and trial ID; outside a trial it uses the supplied name and a generated ID. Its `.name` is the output path and `.final_out` records epoch dictionaries after training begins.

ExperimentLog fixes its CSV metric columns from the first epoch. Subsequent new keys are not added to that file's header. Output files are appended rather than rewritten, and incompatible permissions or filesystem failures propagate. Matplotlib import and display failures arise when plotting begins; missing framework imports arise when constructing the callback.

For native Torch loops, use `ExperimentLog(..., backend='torch')` and invoke the training/epoch hooks from the loop. TrainingPlot's automatic lifecycle is a Keras callback contract. [Analyze](Analyze.md) reads committed trial results; reconstruct a file-based analyzer to see later results.

## Read next

See [energy draw](Energy_Draw.md) for sampled GPU power, [local strategy](Local_Strategy.md) for live file controls, or [Analyze](Analyze.md) for completed results.
