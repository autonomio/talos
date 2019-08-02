# Custom Reducer

A custom reduction strategy can be created and dropped into Talos. Read more about the reduction principle

There are only two criteria to meet:

- The input of the custom strategy is 2-dimensional
- The output of the custom strategy is in the form:

```python
return label, value
```
Here `value` is any hyperparameter value, and `label` is the name of any hyperparameter. Any arbitrary strategy can be implemented, as long as the input and output criteria are met.

The file containing the strategy can then be placed in `/reducers` in Talos package, and corresponding changes made into `/reducers/reduce_run.py` to make the strategy available in `Scan()`. Having done this, the reduction strategy is now available as per the example [above](#probabilistic-reduction).

A [pull request](https://github.com/autonomio/talos/pulls) is highly encouraged once a beneficial reduction strategy has been successfully added.
