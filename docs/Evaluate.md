# Evaluate()

Once the `Scan()` experiment procedures have been completed, the resulting class object can be used as input for `Evaluate()` in order to evaluate one or more models.

```python
from talos import Evaluate

# create the evaluate object
e = Evaluate(scan_object)

# perform the evaluation
e.evaluate(x, y, average='macro')
```

NOTE: It's very important to save part of your data for evaluation, and keep it completely separated from the data you use for the actual experiment. A good approach would be where 50% of the data is saved for evaluation.

### Evaluate Properties

`Evaluate()` has just one property, **`evaluate`**, which is used for evaluating one or more models.

### Evaluate.evaluate Arguments

Parameter | Default | Description
--------- | ------- | -----------
`x` | NA | the predictor data x
`y` | NA | the prediction data y (truth)
`model_id` | None | the model_id to be used
`folds` | None | number of folds to be used for cross-validation
`shuffle` | None | if data is shuffled before splitting
`average` | 'binary' | 'binary', 'micro', 'macro', 'samples', or 'weighted'
`metric` | None | the metric against which the validation is performed
`asc` | None | should be True if metric is a loss

The above arguments are for the <code>evaluate</code> attribute of the <code>Evaluate</code> object.
