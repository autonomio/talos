# Probabilistic Reduction

Based on `Scan()` arguments (listed below), probabilistic reducers drop permutations from the remaining parameter space based on model performance. The process mimics the human process where between experiments the researcher uses various probabilistic methods, such as correlations, to identify how to change the parameter dictionary. When set correctly, reducers can dramatically reduce experiment time, with equal or superior results in comparison to random search.

Argument | Input | Description
-------- | ----- | -----------
`reduction_method` | str | Type of reduction optimizer to be used used
`reduction_interval` | int | Number of permutations after which reduction is applied
`reduction_window` | int | the look-back window for reduction process
`reduction_threshold` | float | The threshold at which reduction is applied
`reduction_metric` | str | The metric to be used for reduction
`minimize_loss` | bool | `reduction_metric` is a loss

The reduction arguments are always invoked through `Scan()`. A typical case involves something on the lines of the below example.

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
Here correlation reducer is used every 2 completed permutations, looking back 2 permutations, and ignoring any correlation that is below 0.2. The reduction is performed with the goal of minimizing validation loss.

In practice this means that Talos will find the hyperparameter with the strongest correlation with high validation loss (undesirable) and then find which individual value has the highest correlation within the identified hyperparameter. Once found, assuming the correlation is higher than `reduction_threshold`, all permutations in the parameter space with that value of the given hyperparameter are dropped.

### Available Reducers

Choose one of the below in `reduction_method`:

- `correlation` (same as `spearman`)
- `spearman`
- `kendall`
- `pearson`
- `trees`
- `forrest`
- `local_strategy` # allows dynamically changing local strategy

For meaningful statistical reduction, use a larger sweep/window than this bounded API example. The native SFD core also provides Sanity, Saturation, Correlation, Focus and Budget reducers; see [SFD and CLI](SFD_and_CLI.md).
