# Gamify

When `Scan(...reduction_method='gamify'...)` between each permutation, a json file is updated on the local machine in the experiment folder. Gamify is an effort to bring the human back on the drive's seat, and is the work of early computer scientists in the late 1950s around the topic of Man-Machine symbiosis.

Gamify allows visualization and two way interaction through two components:

- A browser-based live dashboard
- a round-by-round updating log of each parameter value

### Gamify Dashboard

The historical browser dashboard is an optional external project. Its installation and startup commands depend on the supported dashboard release and your environment; Talos does not require it. See [the dashboard project](https://github.com/autonomio/gamify) for current instructions. The local JSON control below runs without that service.

### Gamify JSON

The JSON file stores the current activity status of each parameter value, and if the status is `active` then nothing will be changed. If the status is `disabled`, then all permutations with that parameter value will be removed from the parameter space.

There is also a numeric value for each parameter value, which is a placeholder for storing an arbitrary value associated with the performance of the parameter value.

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
