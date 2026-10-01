# Local strategy

Use `Scan(reduction_method="local_strategy")` to control pending work and live Scan settings between trials. `local_strategy` is similar to [custom reducers](Custom_Reducers.md) with the difference that the optimization strategy can be changed during the experiment, as the function resides on the local machine.

Define a Python function named `talos_strategy(scan)` in `talos_strategy.py` in the experiment’s working directory. Talos rereads this source between trials and loads a new revision when its content hash changes. A change cannot interrupt the callback already training.

## Prerequisites and procedure

Install the callback’s [backend](Backends.md), use a writable experiment directory and prepare the [Scan minimal example](Scan.md#minimal-example). The strategy executes Python in your experiment process; review it as part of the model code.

1. Create `talos_strategy.py` with `talos_strategy(scan)` before starting the sweep.
2. Set `reduction_method="local_strategy"` in the Scan configuration.
3. Edit the local file between trials when the control needs to change.
4. Inspect the source revisions and control changes recorded in the run audit.

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

## Expected result and failure boundaries

The fragment creates the control file and sets `local_scan.reduction_threshold` to `0.3` after the strategy runs. It does not itself remove candidates. `local_strategy` runs between completed trials independently of the probabilistic reducer interval; use the [parameter-space removal methods](Custom_Reducers.md) for an actual queue decision.

A missing `talos_strategy.py` raises `FileNotFoundError`. A file without callable `talos_strategy(scan)` raises `TypeError`; Python syntax and execution errors propagate. The function may return the Scan context or `None` after mutating it.

On resume, saved source, controls and supported dense data must satisfy the recovery contract. Replacing live data with an opaque stream or a callback that cannot be reconstructed creates a portability boundary; see [SFD and CLI](SFD_and_CLI.md).

## Read next

Use [Gamify](Gamify.md) for parameter-status edits through JSON, or inspect [custom reducers](Custom_Reducers.md) for explicit removal semantics.
