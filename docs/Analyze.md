# Analyze

`talos.Analyze` reads experiment results for metric summaries, parameter comparisons and plots. `talos.Reporting` remains an alias for the same class. Import it from `talos`; training and model selection belong to [Scan](Scan.md) and [Predict](Predict.md).

Analysis may run after Scan completes or from a different shell or kernel while an experiment runs. A file-based instance reads a snapshot; construct it again to see subsequently written trials. Plot methods require the `plots` extra.

## Example

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
r = talos.Analyze(scan_object.run_dir / 'results.csv')

# returns the results dataframe
r.data

# returns the highest value for 'val_accuracy'
r.high('val_accuracy')

# returns the number of rounds it took to find best model
r.rounds2high('val_accuracy')

# draws a histogram for 'val_accuracy'
r.plot_hist('val_accuracy')
```

Reporting works by loading the experiment log .csv file which is saved locally as part of the experiment. Use `experiment_name` for the logging folder or `experiment_dir` for an explicit run directory; the results file is `scan_object.run_dir / "results.csv"`.

## Interface

The signature is `Analyze(source=None)`. Supply a `Scan` or `RunResult` object, a pandas DataFrame, or a CSV path. Although the signature defaults to `None`, it is not a usable source. CSV loading also reads adjacent `metadata.json` when present to identify parameter columns and their aliases. File and CSV parsing errors propagate.

`data` is the result DataFrame. Passing a DataFrame or run object references its current table; passing a path reads a new table. Summary methods do not train models or modify experiment artifacts.

The `Analyze` class object contains several useful properties.

## Results and plots

See docstrings for each function for a more detailed description.

**`high`** The highest result for a given metric

**`rounds`**  The number of rounds in the experiment

**`rounds2high`** The row index label of the highest result; with the standard zero-based result index, it is not a one-based count of trials

**`low`** The lowest result for a given metric

**`correlate`** A pandas Series of numeric-column correlations against a metric; the default method is Pearson, with `spearman` and `kendall` available through `method`

**`plot_line`** A round-by-round line graph for a given metric

**`plot_hist`** A histogram for a given metric where each observation is a permutation

**`plot_corr`** A correlation heatmap where a single metric is compared against hyperparameters

**`plot_regs`** A regression plot with data on two axis

**`plot_box`** A box plot with data on two axis

**`plot_bars`** A bar chart that allows up to 4 axis of data to be shown at once

**`plot_kde`** Kernel Density Estimation type histogram with support for 1 or 2 axis of data

**`table`** A sortable dataframe with a given metric and hyperparameters

**`best_params`** An array of selected parameter values and their rank; use `n=1` for the best model

## Method contracts

| Method | Defaults and result |
| --- | --- |
| `high(metric)`, `low(metric)` | Maximum or minimum value of the named result column. |
| `rounds()` | Number of rows in `data`. |
| `rounds2high(metric)` | Index label returned by pandas `idxmax()`. |
| `correlate(metric, exclude, method='pearson')` | Numeric correlations after excluding named columns; returns a Series without the metric itself. |
| `table(metric, exclude=None, sort_by=None, ascending=False)` | DataFrame sorted by `sort_by`, or `metric` when omitted. `metric` may be a name or list of names. |
| `best_params(metric, exclude, n=10, ascending=False)` | NumPy array of selected parameter values with a final zero-based `index_num` rank column. Use `ascending=True` for loss. |
| `plot_line(metric)`, `plot_hist(metric, bins=10)` | Matplotlib Axes; one point or histogram observation per result row. |
| `plot_corr(metric, exclude, color_grades=5)` | Axes for a numeric-column correlation heatmap. |
| `plot_regs(x, y)`, `plot_box(x, y, hue=None)` | Axes for a regression scatter plot or grouped box plot. |
| `plot_bars(x, y, hue, col)` | Matplotlib Figure with one subplot for each `col` group. |
| `plot_kde(x, y=None)` | Axes for one-dimensional density, or a bivariate density when `y` is supplied. |

Result columns must exist. Missing names raise pandas `KeyError`; correlation requires a numeric metric. Bivariate KDE requires at least three finite pairs with variation on both axes; singular distributions can still fail density estimation. Plotting allocates figures in the caller's Matplotlib environment.

For Talos 2 runs, `best_params()` uses recorded parameter aliases so a hyperparameter named `loss` remains distinct from the measured loss. A standalone historical CSV without metadata cannot recover that distinction automatically; its `best_params()` output includes other nonexcluded result columns as well as parameters.

## Read next

Use [Predict](Predict.md) to select a model, [Evaluate](Evaluate.md) to score it on held-out observations, or [Deploy](Deploy.md) to package the selected model.
