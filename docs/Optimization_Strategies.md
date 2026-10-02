# Optimization strategies

Talos supports several common optimization strategies:

- Grid search
- Random search
- [Probabilistic reduction](Probabilistic_Reduction.md)
- [Custom reducers](Custom_Reducers.md)
- [Local strategies](Local_Strategy.md)
- [Gamify](Gamify.md) parameter controls

The search selects model configurations: each completed trial trains one candidate combination from the declared parameter space.

The established Talos search features emphasize:

- adding variations of random variable picking
- reducing the workload of random variable picking

Sampling chooses a bounded candidate set; reduction removes remaining candidates using completed metrics or explicit controls. Native SFD search strategies are described in [SFD and CLI](SFD_and_CLI.md).

## Before choosing a strategy

Install the [backend](Backends.md) used by the training callback and execute the [Scan minimal example](Scan.md#minimal-example), which supplies the arrays and model used below. Start with a writable experiment directory. External entropy methods require caller-owned provider credentials and a usable provider service; see [remote entropy configuration](#remote-entropy-configuration).

1. Determine the full candidate count from the declared lists or expanded ranges.
2. Choose a full grid or a sampling limit and method.
3. Choose the metric and direction before adding a reducer; changing the method does not change the meaning of the model’s metric.
4. Run the bounded fragment below and inspect the selected parameter values in the results table.

## Random search

Talos supports three categories of sampling sources:

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

### Random options

PARAMETER | DESCRIPTION
--------- | -----------
`ambience` | External RANDOM.ORG entropy service
`halton` | Halton sequences
`korobov_matrix` | Korobov matrix based sequence
`latin_matrix` | Latin hypercube
`latin_improved` | Improved Latin hypercube
`latin_sudoku` | Latin hypercube with a Sudoku-style constraint
`quantum` | External quantum entropy service
`sobol` | Sobol sequences
`uniform_crypto` | Cryptographically sound uniform
`uniform_mersenne` | Uniform Mersenne twister

Each method differs in discrepancy and other observable aspects. `seed` makes supported offline methods repeatable; `uniform_crypto`, `quantum`, and `ambience` draw fresh entropy. Checkpoints retain the realized pending rows so a resumed sweep does not resample them. Performance depends on the search space; compare methods on your own experiment. `quantum` and `ambience` use the provider configuration below; availability, cost and physical entropy claims are service-specific. The default workflow and examples use offline methods.

### Remote entropy configuration

Talos owns both HTTPS clients. `quantum` uses the current
[ANU Quantum Numbers API](https://quantumnumbers.anu.edu.au/documentation);
`ambience` uses [RANDOM.ORG's current JSON-RPC API](https://api.random.org/json-rpc/4/basic).
Obtain a key from the selected provider and save it as one UTF-8 line in a
caller-owned file outside the project, manifest store and result directories.
Set `TALOS_ANU_KEY_FILE` or `TALOS_RANDOM_ORG_KEY_FILE` to that file's path.
RANDOM.ORG keys use UUID syntax. Restrict file access to the account running
Talos. Replacing the file's contents rotates the credential; each service
request reads it again.

ANU requests batches of 1,024 unsigned 16-bit values and retains Talos's legacy
conversion to candidate indexes. RANDOM.ORG requests integer sequences in
batches of at most 10,000, with maximum indexes up to 1,000,000,000. Talos
validates response types, lengths, bounds and identities, then retains the
existing selection of unique indexes in the legal parameter space. Its
32-attempt refill bound still fails visibly when a provider cannot supply the
requested population.

Both clients verify certificate chains and hostnames, require TLS 1.2 or newer,
refuse redirects and apply a 15-second transport timeout. Credentials stay out
of URLs and Talos logs. Missing or malformed keys, certificate failures,
invalid responses and exhausted quotas fail the selected sampler. RANDOM.ORG's
reported advisory delay is honored up to 60 seconds; a larger or invalid delay
fails instead of blocking the run indefinitely.

Loopback tests verify these transport and protocol contracts with controlled
responses. They do not establish live account access, service availability or
physical entropy quality.

For `Scan(..., params=dict, resume=True, experiment_dir=...)` without an explicit
`search_strategy`, Talos reconstructs its original realized rows from saved run
metadata before sampling or executing
`boolean_limit`. The normal runner then validates the current parameters,
model/preparation source, data, seed and environment before restoring the
checkpoint's remaining queue. Provider availability and new entropy do not
change recovery. Sampling limits and the boolean selection are frozen to those
original rows during dictionary recovery; changing them does not regenerate
candidates. A caller-created `ParamSpace` or explicit native `search_strategy`
remains caller-owned and uses the runner's existing checkpoint recovery path.

New dictionary runs record their ordered parameter keys in the hashed identity;
reordered caller keys fail before saved rows are reused. The older record format
without that witness remains supported and requires the caller to preserve its
original dictionary key order; those records cannot universally attest the
order. Resume still requires the recorded Talos version, backend/dependency
environment and caller source identities. Record-format compatibility does not
migrate checkpoints across versions. Missing or malformed original row state
fails without resampling.

## Grid search

For conventional grid search, omit both `fraction_limit` and `round_limit`. The parameter space enumerates all candidate combinations; `boolean_limit`, reducers, `performance_target`, `time_limit` and native stop controls can still reduce the work completed.

## Early stopping

A backend early-stopping callback can stop one candidate’s training when the monitored epoch metric no longer improves. The sweep then proceeds to the next candidate. Talos provides `lazy`, `moderate` and `strict` presets as well as custom settings. This differs from reducing the pending parameter space.

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

## Expected result and failure boundaries

The seeded sampling example completes half of the Iris setup’s parameter combinations. `random_scan.data` shows the combinations actually trained, and checkpoints retain the realized pending rows. The early-stopping fragment returns a new Keras training history; it does not remove other candidates from the sweep.

A sampling limit that produces fewer than one candidate raises a data error. Unknown random method names raise an error when sampling is requested. External entropy providers can fail or return insufficient unique indices; use an offline method for a provider-independent run. A seed does not establish deterministic device kernels or repeat fresh cryptographic/service entropy.

Reducers can end a sweep before the requested count is reached. For meaningful statistical decisions, use enough completed trials and inspect the run audit; a small documentation fixture establishes the interface only.

## Read next

Configure [probabilistic reduction](Probabilistic_Reduction.md), write a [custom reducer](Custom_Reducers.md), or inspect the native search and recovery surfaces in [SFD and CLI](SFD_and_CLI.md).
