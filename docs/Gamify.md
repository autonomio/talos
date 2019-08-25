# Gamify

When `Scan(...reduction_method='gamify'...)` between each permutation, a json file is updated on the local machine in the experiment folder. Gamify is an effort to bring the human back on the drive's seat, and is the work of early computer scientists in the late 1950s around the topic of Man-Machine symbiosis.

Gamify allows visualization and two way interaction through two components:

- A browser-based live dashboard
- a round-by-round updating log of each parater value

### Gamify Dasbhoard

First install the add-on package:

`pip install gamify`

Then add `gamify` to your `PATH`:

`export PATH=$PATH:/Users/mikko/miniconda3/envs/wip_conda/lib/python3.6/site-packages/gamify.py`

Note that your path will be unique to your machine, you can see it on the terminal output once you install the package.

Now you can start the dashboard by referencing an experiment folder.

`python gamify /path/to/talos/experiment`

See more information [here](https://github.com/autonomio/gamify)

### Gamify JSON

The JSON file stores the current activity status of each parameter value, and if the status is `active` then nothing will be changed. If the status is `disabled`, then all permutations with that parameter value will be removed from the parameter space.

There is also a numeric value for each parameter value, which is a placeholder for storing an arbitrary value associated with the performance of the parameter value.
