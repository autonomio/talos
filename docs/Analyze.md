# Analyze (previously Reporting)

The experiment results can be analyzed through the [Analyze()](https://github.com/autonomio/talos/blob/master/talos/commands/analyze.py) utility. `Analyze()` may be used after Scan completes, or during an experiment (from a different shell / kernel).

## Analyze Use

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

## Analyze Arguments

`Analyze()` has only a single argument `source`. This can be either a .csv file which results `Scan()` or the class object which also results from `Scan()`.

The `Analyze` class object contains several useful properties.

## Analyze Properties

See docstrings for each function for a more detailed description.

**`high`** The highest result for a given metric

**`rounds`**  The number of rounds in the experiment

**`rounds2high`** The number of rounds it took to get highest result

**`low`** The lowest result for a given metric

**`correlate`** A dataframe with Spearman correlation against a given metric

**`plot_line`** A round-by-round line graph for a given metric

**`plot_hist`** A histogram for a given metric where each observation is a permutation

**`plot_corr`** A correlation heatmap where a single metric is compared against hyperparameters

**`plot_regs`** A regression plot with data on two axis

**`plot_box`** A box plot with data on two axis

**`plot_bars`** A bar chart that allows up to 4 axis of data to be shown at once

**`plot_kde`** Kernel Density Estimation type histogram with support for 1 or 2 axis of data

**`table`** A sortable dataframe with a given metric and hyperparameters

**`best_params`** An array of selected parameter values and their rank; use `n=1` for the best model
