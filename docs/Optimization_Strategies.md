# Optimization Strategies

Talos supports several common optimization strategies:

- Grid search
- Random search
- Probabilistic reduction
- Custom Strategies

The object of abstraction is the keras model configuration, of which n number of permutations is tried in a Talos experiment.

As opposed to adding more complex optimization strategies, which are widely available in various solutions, Talos focus is on:

- adding variations of random variable picking
- reducing the workload of random variable picking

As it stands, both of these approaches are currently under leveraged by other solutions, and under represented in the literature.

# Random Search

A key focus in Talos develoment is to provide gold standard random search capabilities. Talos implements three kinds of random generation methods:

- True / Quantum randomness
- Pseudo randomness
- Quasi randomness

Random methods are selected in `Scan(...random_method...)`. For example, to use `quantum` randomness:

```python
talos.Scan(x=x, y=y, model=model, params=p, random_method='quantum')
```

## Random Options

PARAMETER | DESCRIPTION
--------- | -----------
`ambience` | Ambient Sound based randomness
`halton` | Halton sequences
`korobov_matrix` | Korobob matrix based sequence
`latin_matrix` | Latin hypercube
`latin_improved` | Improved Latin hypercube
`latin_sudoku` | Latin hypercube with a Sudoku-style constraint
`quantum` | Quantum randomness (vacuum based)
`sobol` | Sobol sequences
`uniform_crypto` | Cryptographically sound uniform
`uniform_mersenne` | Uniform Mersenne twister

Each method differs in discrepancy and other observable aspects. Scientific evidence suggest that low discrepancy methods outperform plain pseudo-random methods.

# Grid Search

To perform a conventional grid search, simply leave the `Scan(...fraction_limit...)` argument undeclared, that way all possible permutations will be processed in a sequential order.

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

```python
talos.Scan(...
          reduction_method='correlation',
          reduction_interval=50,
          reduction_window=25,
          reduction_threshold=0.2,
          reduction_metric='mae',
          minimize_loss=True,
          ...)
```
Here correlation reducer is used every 50 permutations, looking back 25 permutations, and ignoring any correlation that is below 0.2. The reduction is performed with the goal of minimizing mae.

In practice this means that Talos will find the hyperparameter with the strongest correlation with high mae (undesirable) and then find which individual value has the highest correlation within the identified hyperparameter. Once found, assuming the correlation is higher than `reduction_threshold`, all permutations in the parameter space with that value of the given hyperparameter are dropped.

# Early Stopping

Use of early stopper, when set appropriate, can help reduce experiment time by preventing time waste on unproductive permutations. Once a monitored metric is no longer improving, Talos moves to the next permutation. Talos provides three presets - `lazy`, `moderate` and `strict` - in addition to completely custom settings.

`early_stopper` is invoked in the input model, in `model.fit()`.

```python

out = model.fit(x_train,
                y_train,
                batch_size=params['batch_size'],
                epochs=params['epochs'],
                validation_data=[x_val, y_val],
                verbose=0,
                callbacks=[talos.utils.early_stopper(params['epochs'])])

```

The minimum input to Talos `early_stopper` is the `epochs` hyperparameter. This is used with the automated settings. For custom settings, this can be left as `None`.

Argument | Input | Description
-------- | ----- | -----------
`epochs` | int | The number of epochs for the permutation e.g. params['epochs']
`monitor` | int | The metric to monitor for change
`mode` | str | One of the presets `lazy`, `moderate`, `strict` or `None`
`min_delta` | float | The limit for change at which point flag is raised
`patience` | str | the number of epochs before termination from flag

# Custom Reducer

A custom reduction strategy can be created and dropped into Talos. Read more about the reduction principle

There are only two criteria to meet:

- The input of the custom strategy is 2-dimensional
- The output of the custom strategy is in the form:

```python
return value, label
```
Here `value` is any hyperparameter value, and `label` is the name of any hyperparameter. Any arbitrary strategy can be implemented, as long as the input and output criteria are met.

The file containing the strategy can then be placed in `/reducers` in Talos package, and corresponding changes made into `/reducers/reduce_run.py` to make the strategy available in `Scan()`. Having done this, the reduction strategy is now available as per the example [above](#probabilistic-reduction).

A [pull request](https://github.com/autonomio/talos/pulls) is highly encouraged once a beneficial reduction strategy has been successfully added.
