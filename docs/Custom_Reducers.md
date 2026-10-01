# Custom reducers

Pass a Python callable to `Scan(reduction_method=…)` to remove pending candidates using your own decision rule. The reducer receives the live Scan context and runs between trials; completed results remain recorded.

## Prerequisites

Install the callback’s [backend](Backends.md), run the [Scan minimal example](Scan.md#minimal-example), and choose a decision based on recorded trial metrics or explicit experiment knowledge. The working directory must be writable for scan artifacts. Read [probabilistic reduction](Probabilistic_Reduction.md) for the shared pending-work boundary.

## Procedure

1. Define the reducer below using the live Scan context.
2. Return the original parameter name and candidate value to remove, or update `scan.param_object` directly.
3. Pass the callable through `reduction_method` and choose its invocation interval.
4. Inspect the completed results and pending selection in the run audit.

The callable contract has two parts:

- The input is the live Scan context; `scan.data` is a 2-dimensional results table
- The output of the custom strategy is in the form:

```python
def custom_reducer(scan):
    # Return an original parameter name and candidate value to remove.
    label, value = 'first_neuron', 8
    return label, value
```

Here `label` is an original parameter name and `value` is a candidate from that parameter. Returning this pair invokes `remove_is(label, value)` on pending rows. The example deliberately removes `first_neuron=8`; replace that rule with a decision appropriate for your experiment.

With these in place, one then proceeds to apply the reduction to the current parameter space, with one of the supported functions:

| Method on `scan.param_object` | Pending-work effect |
|---|---|
| `remove_is_not(label, value)` | Retain rows whose named parameter equals the value |
| `remove_is(label, value)` | Remove rows whose named parameter equals the value |
| `remove_le(label, value)` | Remove rows at or below the threshold |
| `remove_ge(label, value)` | Remove rows at or above the threshold |
| `remove_lambda(predicate)` | Retain rows for which the row-dictionary predicate returns `True` |

See the [built-in correlation implementation](../talos/reducers/correlation.py) to make sure you understand the expected structure of a custom reducer.

Pass the function directly through `reduction_method`; editing the installed Talos package is unnecessary. A custom strategy may instead mutate `scan.param_object` and return the Scan context. Predicates passed to `remove_lambda` return `True` for rows to retain.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
custom_scan = talos.Scan(x, y, p, input_model, 'custom', x_val=x_val, y_val=y_val,
                         reduction_method=custom_reducer, reduction_interval=1,
                         seed=17, disable_progress_bar=True)
```

## Expected result and failure boundaries

`custom_scan.data` retains completed trial rows. With the linked Iris setup, the first completed row uses `first_neuron=4`; the reducer removes the pending `8` candidate, so the sweep can complete fewer rows than its original parameter count. Removal affects pending work and does not erase observations.

Unknown parameter names and unsuitable comparisons can fail inside the reducer. A callable may return the modified Scan context or `None` after mutating it; a pair must name the value to remove. Keep the reducer importable when its callable identity needs to survive resume. Exceptions in custom Python control surface through the run rather than becoming successful trial results.

## Read next

Use [local strategy](Local_Strategy.md) when the control source must change during the run. Review [contribution guidance](../CONTRIBUTING.md) when proposing a reusable reducer.
