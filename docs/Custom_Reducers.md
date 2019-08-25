# Custom Reducer

A custom reduction strategy can be created and dropped into Talos. Read more about the reduction principle

There are only two criteria to meet:

- The input of the custom strategy is 2-dimensional
- The output of the custom strategy is in the form:

```python
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

The file containing the custom strategy can then be placed in `/reducers` in Talos package, and corresponding changes made into `/reducers/reduce_run.py` to make the strategy available in `Scan()`. Having done this, the reduction strategy is now available as per the example [above](#probabilistic-reduction).

A [pull request](https://github.com/autonomio/talos/pulls) is highly encouraged once a beneficial reduction strategy has been successfully added.
