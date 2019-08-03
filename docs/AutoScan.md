# AutoScan

`AutoScan()` provides a streamlined way for conducting a hyperparameter search experiment with any dataset. It is particularly useful for early exploration as with default settings `AutoScan()` casts a very broad parameter space including all common hyperparameters, network shapes, sizes, as well as architectures

Configure the `AutoScan()` experiment and then use the property `start` in the returned class object to start the actual experiment.

```python
auto = talos.autom8.AutoScan(task='binary', max_param_values=2)
auto.start(x, y, experiment_name='testing.new', fraction_limit=0.001)
```

NOTE: `auto.start()` accepts all `Scan()` arguments.

## AutoScan Arguments

Argument | Input | Description
--------- | ------- | -----------
`task` | str or None | `binary`, `multi_label`, `multi_class`, or `continuous`
`max_param_values` | int | Number of parameter values to be included

Setting `task` effects which various aspects of the model and should be set according to the specific prediction task, or set to `None` in which case `metric` input is required.

## AutoScan Properties

The only property **`start`** starts the actual experiment. `AutoScan.start()` accepts the following arguments:

Argument | Input | Description
--------- | ------- | -----------
`x` | array or list of arrays | prediction features
`y` | array or list of arrays | prediction outcome variable
`kwargs` | arguments | any `Scan()` argument can be passed into `AutoScan.start()`
