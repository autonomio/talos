# Local Strategy

With `Scan(...reduction_method='local_strategy'...)` it's possible to control the experiment between each permutation without interrupting the experiment. `local_strategy` is similar to [custom strategies](#custom-reducers) with the difference that the optimization strategy can be changed during the experiment, as the function resides on the local machine.

The `talos_strategy` should be a python function stored in a file `talos_strategy.py` which resides in the present working directory of the experiment. It can be changed anytime.
