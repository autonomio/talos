# Probabilistic reduction

Probabilistic reducers drop pending parameter combinations using completed model metrics. They formalize the research step of inspecting past results and deciding which values to stop trying. They can reduce completed work; whether that improves the experiment depends on the metric, sample size and search space.

Argument | Input | Description
-------- | ----- | -----------
`reduction_method` | str or callable | Reducer name or callable
`reduction_interval` | int | Number of permutations after which reduction is applied
`reduction_window` | int | the look-back window for reduction process
`reduction_threshold` | float | The threshold at which reduction is applied
`reduction_metric` | str | The metric to be used for reduction
`minimize_loss` | bool | `reduction_metric` is a loss

## Prerequisites and procedure

Install the callback’s [backend](Backends.md) and execute the [Scan minimal example](Scan.md#minimal-example). Choose a recorded numeric metric and its direction before configuring reduction. A writable run directory retains result and audit artifacts.

1. Select a reducer and a metric present in the callback history.
2. Set the interval and look-back window according to the completed observations needed for a useful decision.
3. Run the bounded API example below.
4. Inspect completed rows and the remaining queue before scaling to a research experiment.

The reduction arguments are passed to `Scan()`.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
reduced = talos.Scan(x, y, p, input_model, 'reduced',
          x_val=x_val, y_val=y_val, seed=17, disable_progress_bar=True,
          reduction_method='correlation',
          reduction_interval=2,
          reduction_window=2,
          reduction_threshold=0.2,
          reduction_metric='val_loss',
          minimize_loss=True)
```

The example configures a Spearman decision every two completed trials, with a two-trial look-back and threshold `0.2`, to minimize validation loss. The correlation implementation requires at least three finite target observations and variation in the target, so this deliberately small window does not prune. Increase the window and search size for a statistical experiment.

For each parameter value observed in the window, Talos builds a presence indicator and correlates it with the selected metric. The strongest unfavorable finite correlation meeting `reduction_threshold` selects a value to remove from pending combinations. When minimizing loss, a positive association with higher loss is unfavorable; when maximizing a metric, a negative association is unfavorable. Completed trials remain recorded.

## Available reducers

Choose one of the below in `reduction_method`:

- `correlation` (same as `spearman`)
- `spearman`
- `kendall`
- `pearson`
- `trees`
- `forrest`
- `local_strategy`: dynamically changing local strategy; see [local strategy](Local_Strategy.md)

For meaningful statistical reduction, use a larger sweep/window than this bounded API example. The native SFD core also provides Sanity, Saturation, Correlation, Focus and Budget reducers; see [SFD and CLI](SFD_and_CLI.md).

## Expected result and failure boundaries

`reduced.data` contains completed metrics and candidate values; `reduced.param_object.param_index` represents pending legacy rows. A pruning decision reduces the pending set and enters the run audit. Too few observations, a constant target or no varying candidate indicators produce no useful statistical reduction.

A metric absent from the result table raises an error. Choose `minimize_loss` to match the metric’s meaning: the name alone does not decide its direction. Tree reducers use a different decision mechanism; `reduction_threshold` governs the correlation family rather than every reducer. The accepted forest reducer name retains the historical spelling `forrest`.

## Read next

Use [custom reducers](Custom_Reducers.md) for an experiment-specific decision, [local strategy](Local_Strategy.md) for live Python control, or [SFD and CLI](SFD_and_CLI.md) for native reducer objects and recovery artifacts.
