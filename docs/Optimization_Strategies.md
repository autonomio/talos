# Optimization Strategies

Talos supports several common optimization strategies:

- Grid search
- Random search
- Probabilistic reduction
- Custom Strategies (arbitrary single python file optimizer)
- Local Stragies (can change anytime during experiment)
- Gamify (man-machine cooperation)

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
