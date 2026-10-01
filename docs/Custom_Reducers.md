# Custom Reducer

A custom reduction strategy can be created and dropped into Talos. Read more about the reduction principle

There are only two criteria to meet:

- The input is the live Scan context; `scan.data` is a 2-dimensional results table
- The output of the custom strategy is in the form:

```python
def custom_reducer(scan):
    # Return an original parameter name and candidate value to remove.
    label, value = 'first_neuron', 8
    return label, value
```
Here `value` is any hyperparameter value, and `label` is the name of any hyperparameter. Any arbitrary strategy can be implemented, as long as the input and output criteria are met.

With these in place, one then proceeds to apply the reduction to the current parameter space, with one of the supported functions:

- `remove_is_not`
- `remove_is`
- `remove_le`
- `remove_ge`
- `remove_lambda`

See [a working example](https://github.com/autonomio/talos/blob/master/talos/reducers/correlation.py) to make sure you understand the expected structure of a custom reducer.

Pass the function directly through `reduction_method`; editing the installed Talos package is unnecessary. A custom strategy may instead mutate `scan.param_object` and return the Scan context. Predicates passed to `remove_lambda` return `True` for rows to retain.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
custom_scan = talos.Scan(x, y, p, input_model, 'custom', x_val=x_val, y_val=y_val,
                         reduction_method=custom_reducer, reduction_interval=1,
                         seed=17, disable_progress_bar=True)
```

A [pull request](https://github.com/autonomio/talos/pulls) is highly encouraged once a beneficial reduction strategy has been successfully added.
