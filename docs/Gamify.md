# Gamify

Use `Scan(reduction_method="gamify")` to inspect and change parameter-value statuses through a local JSON file between trials. This preserves the original human-in-the-loop search pattern: a researcher can remove pending values after observing completed results.

The historical Gamify design has two components:

- an optional external browser dashboard
- a round-by-round updating log of each parameter value

## Dashboard boundary

The historical browser dashboard is an optional external project. Its installation and startup commands depend on the supported dashboard release and your environment; Talos does not require it. See [the dashboard project](https://github.com/autonomio/gamify) for current instructions. The local JSON control below runs without that service.

## JSON control

The JSON file stores the current activity status of each parameter value, and if the status is `active` then nothing will be changed. If the status is `disabled`, then all permutations with that parameter value will be removed from the parameter space.

There is also a numeric value for each parameter value, which is a placeholder for storing an arbitrary value associated with the performance of the parameter value.

## Prerequisites and procedure

Install the callback’s [backend](Backends.md), prepare the [Scan minimal example](Scan.md#minimal-example) and use a writable experiment directory. The local JSON interface requires no dashboard service.

1. Start a scan with `reduction_method="gamify"` and pause after one trial using `stop_after=1`.
2. Read the generated control file and change one candidate’s status to `disabled`.
3. Resume the same run directory with unchanged model, data and configuration identity.
4. Inspect the retained completed rows and removed pending candidates.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

The JSON remains beside the legacy experiment run under the `experiment_name` folder. Status changes remove pending candidates; numeric annotations alone do not prune.

```python
from pathlib import Path
import json

gamify_scan = talos.Scan(x, y, p, input_model, 'gamify_demo', x_val=x_val, y_val=y_val,
                        reduction_method='gamify', stop_after=1, seed=17,
                        disable_progress_bar=True)
control_path = Path(gamify_scan.experiment_name) / (gamify_scan.run_dir.name + '.json')
control = json.loads(control_path.read_text())
control['0']['1'][0] = 'disabled'
control['0']['1'][1] = .75
control_path.write_text(json.dumps(control))
gamify_resumed = talos.Scan(x, y, p, input_model, 'gamify_demo', x_val=x_val, y_val=y_val,
                           reduction_method='gamify', experiment_dir=gamify_scan.run_dir,
                           resume=True, seed=17, disable_progress_bar=True)
```

## Expected result and failure boundaries

The fragment keeps the first completed Iris trial and disables the second `first_neuron` candidate before resume schedules it. `gamify_resumed.data` includes the retained completed result. The control file stays at `experiment_name/<run-directory-name>.json`; its nested numeric string keys follow parameter and candidate order. The numeric `.75` annotation is recorded but does not make a pruning decision.

`disabled` removes matching pending rows. Changing a value back to `active` does not reconstruct already removed queue entries or undo completed work. Keep the generated JSON structure intact; malformed JSON, missing candidate entries and incompatible edits can fail when Talos reads the control file. Use an atomic file replacement when an external editor writes while the scan is running.

The optional dashboard has its own maintenance and installation boundary. JSON control, checkpoints and audit behavior belong to Talos.

## Read next

Use [local strategy](Local_Strategy.md) for live Python controls or [SFD and CLI](SFD_and_CLI.md) for native interventions and resume inspection.
