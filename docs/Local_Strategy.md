# Local Strategy

With `Scan(...reduction_method='local_strategy'...)` it's possible to control the experiment between each permutation without interrupting the experiment. `local_strategy` is similar to [custom strategies](#custom-reducers) with the difference that the optimization strategy can be changed during the experiment, as the function resides on the local machine.

The `talos_strategy` should be a python function stored in a file `talos_strategy.py` which resides in the present working directory of the experiment. It can be changed anytime.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

The following fragment writes a local strategy before starting a bounded sweep. Use it in an experiment working directory, since `talos_strategy.py` is a live control file.

```python
from pathlib import Path
Path('talos_strategy.py').write_text(
    "def talos_strategy(scan):\n"
    "    scan.reduction_threshold = .3\n"
    "    return scan\n")
local_scan = talos.Scan(x, y, p, input_model, 'local', x_val=x_val, y_val=y_val,
                       reduction_method='local_strategy', seed=17,
                       disable_progress_bar=True)
```

Source revisions and actual parameter/control changes enter the run audit. Changed model, train/validation arrays and save/print/cleanup controls are preserved at checkpoints. Changed opaque streams or nonimportable replacement callbacks may train in-process but cannot be reconstructed for resume; resume reports that limitation explicitly.
