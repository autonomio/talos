# Deploy()

A successful experiment can be deployed easily. Deploy() takes in the object from Scan() and creates a package locally that can be later activated with Restore().

```python
from talos import Deploy

Deploy(scan_object, 'experiment_name')
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

## Deploy Package Contents

The deploy package consists of:

- details of the scan (details.txt)
- model weights (model.h5)
- model json (model.json)
- results of the experiment (results.csv)
- sample of x data (x.csv)
- sample of y data (y.csv)

The package can be restored into a copy of the original Scan object using the `Restore()` command.
