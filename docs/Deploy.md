# Deploy()

A successful experiment can be deployed easily. Deploy() takes in the object from Scan() and creates a package locally that can be later activated with Restore().

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Deploy

deployment = Deploy(scan_object, 'experiment_name', metric='val_loss', asc=True)
```

When you've achieved a successful result, you can use `Deploy()` to prepare a production ready package that can be easily transferred to another environment or system, or sent or uploaded. The deployment package will consists of the best performing model, which is picked base on the `metric` argument.

NOTE: for a metric that is to be minimized, set `asc=True` or otherwise
you will end up with the model that has the highest loss.

## Deploy Arguments

Parameter | type | Description
--------- | ------- | -----------
`scan_object` | class object | a `Scan` object
`model_name` | str | Name for the .zip file to be created.
`metric` | str | The metric to be used for picking the best model.
`asc` | bool | Make this True for metrics that are to be minimized (e.g. loss)
`saved` | bool | if a model saved on local machine should be used
`custom_objects` | dict | if the model has a custom object, pass it here
`model_factory` | callable or None | Reconstructs a saved Torch model when its architecture cannot be recovered from the archive.

## Deploy Package Contents

The deploy package consists of:

- archive version, selected model and artifact metadata (`manifest.json`)
- backend-native trained model (`model.keras`, Torch state or joblib as appropriate)
- details and epoch histories (`details.json`, `history.json`)
- results of the experiment (`results.csv`)
- original parameters and samples of x/y data (`params.npy`, `x.npy`, `y.npy`)
- verified caller source snapshots when available

The package can be restored into a copy of the original Scan object using the `Restore()` command.

Only restore archives from trusted sources: legacy parameter/sample compatibility uses Python object serialization. This command packages a local archive; it does not publish a service.
