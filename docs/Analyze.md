# Analyze (previously Reporting)

The experiment results can be analyzed through the [Analyze()](https://github.com/autonomio/talos/blob/master/talos/utils/reporting.py) utility. `Analyze()` may be used after Scan completes, or during an experiment (from a different shell / kernel).

## Analyze Use

```python
r = Reporting('experiment_log.csv')

# returns the results dataframe
r.data

# returns the highest value for 'val_fmeasure'
r.high('val_fmeasure')

# returns the number of rounds it took to find best model
r.rounds2high()

# draws a histogram for 'val_acc'
r.plot_hist()
```

Reporting works by loading the experiment log .csv file which is saved locally as part of the experiment. The filename can be changed through dataset_name and experiment_no Scan arguments.

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

**`plot_kde`** Kernel Destiny Estimation type histogram with support for 1 or 2 axis of data

**`table`** A sortable dataframe with a given metric and hyperparameters

**`best_params`** A dictionary of parameters from the best model
