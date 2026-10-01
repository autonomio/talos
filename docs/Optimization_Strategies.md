# Optimization Strategies

Talos supports several common optimization strategies:

- Grid search
- Random search
- Probabilistic reduction
- Custom Strategies (arbitrary single python file optimizer)
- Local Strategies (can change anytime during experiment)
- Gamify (man-machine cooperation)

The object of abstraction is the model configuration, of which n number of permutations is tried in a Talos experiment.

As opposed to adding more complex optimization strategies, which are widely available in various solutions, Talos focus is on:

- adding variations of random variable picking
- reducing the workload of random variable picking

As it stands, both of these approaches are currently under leveraged by other solutions, and under represented in the literature.

# Random Search

A key focus in Talos develoment is to provide gold standard random search capabilities. Talos implements three kinds of random generation methods:

- True / Quantum randomness
- Pseudo randomness
- Quasi randomness

Random methods are selected through `random_method` when a sampling limit is supplied. The following example uses an offline seeded method:

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
random_scan = talos.Scan(x, y, p, input_model, 'random', x_val=x_val, y_val=y_val,
                         random_method='uniform_mersenne', fraction_limit=.5,
                         seed=17, disable_progress_bar=True)
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

Each method differs in discrepancy and other observable aspects. `seed` makes supported offline methods repeatable; `uniform_crypto`, `quantum`, and `ambience` draw fresh entropy. Checkpoints retain the realized pending rows so a resumed sweep does not resample them. Performance depends on the search space; compare methods on your own experiment. `quantum` and `ambience` require the optional `chances` package and external entropy services; availability, cost, and physical entropy claims are service-specific. They are not required for the examples or the default workflow.

# Grid Search

To perform a conventional grid search, simply leave the `Scan(...fraction_limit...)` argument undeclared, that way all possible permutations will be processed in a sequential order.

# Early Stopping

Use of early stopper, when set appropriate, can help reduce experiment time by preventing time waste on unproductive permutations. Once a monitored metric is no longer improving, Talos moves to the next permutation. Talos provides three presets - `lazy`, `moderate` and `strict` - in addition to completely custom settings.

`early_stopper` is invoked in the input model, in `model.fit()`.

```python

_, model = input_model(x, y, x_val, y_val, {key: values[0] for key, values in p.items()})
params = {key: values[0] for key, values in p.items()}
out = model.fit(x,
                y,
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
`monitor` | str | The metric to monitor for change
`mode` | str | One of the presets `lazy`, `moderate`, `strict` or `None`
`min_delta` | float | The limit for change at which point flag is raised
`patience` | int | the number of epochs before termination from flag
