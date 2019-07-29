# Monitoring

There are several options for monitoring the experiment.

```python
# turn off progress bar
Scan(disable_progress_bar=True)

# enable live training plot
from talos import live
out = model.fit(X,
                Y,
                epochs=20,
                callbacks=[live()])

# turn on parameter printing
Scan(print_params=True)
```

**Progress Bar :** A round-by-round updating progress bar that shows the remaining rounds, together with a time estimate to completion. Progress bar is on by default.

**Live Monitoring :** Live monitoring provides an epoch-by-epoch updating line graph that is enabled through the `live()` custom callback.

**Round Hyperparameters :** Displays the hyperparameters for each permutation. Does not work together with live monitoring.
