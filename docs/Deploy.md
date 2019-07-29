## Deploy()

> A successful experiment can be deployed easily

```python
from talos import Deploy

Deploy(scan_object, 'experiment_name')
```

When you've achieved a successful result, you can use `Deploy()` to prepare a production ready package that can be easily transferred to another environment or system, or sent or uploaded. The deployment package will consists of the best performing model, which is picked automatically against 'val_acc' unless stated with `metric` argument.

The deploy package consists of:

- details of the scan (details.txt)
- model weights (model.h5)
- model json (model.json)
- results of the experiment (results.csv)
- sample of x data (x.csv)
- sample of y data (y.csv)

The package can be restored into a copy of the original Scan object using the `Restore()` command.

### Deploy Arguments

Parameter | Default | Description
--------- | ------- | -----------
`scan_object` | None | a `Scan` object
`model_name` | None | a string value as the name of experiment
`metric` | 'val_acc' | metric against which best model is picked
`asc` | False | use True for loss functions
