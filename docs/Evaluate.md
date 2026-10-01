# Evaluate()

Once the `Scan()` experiment procedures have been completed, the resulting class object can be used as input for `Evaluate()` in order to evaluate one or more models.

Examples below use the held-out Iris setup in [Scan → Minimal Example](Scan.md#minimal-example). Run that setup first; it defines `scan_object`, `p`, `input_model`, `x`, `y`, `x_val`, `y_val`, `x_test`, and `y_test`.

```python
from talos import Evaluate

# create the evaluate object
e = Evaluate(scan_object)

# perform the evaluation
scores = e.evaluate(x_test, y_test, task='multi_class', metric='val_loss',
                    asc=True, average='macro', folds=3, seed=17)
```

NOTE: It's very important to save part of your data for evaluation, and keep it completely separated from the data you use for the actual experiment. Choose an evaluation fraction suited to the dataset; the shared example reserves 20%. These folds score the same fitted model on held-out subsets; they do not retrain it.

### Evaluate Properties

`Evaluate()` has just one property, **`evaluate`**, which is used for evaluating one or more models.

### Evaluate.evaluate Arguments

Parameter | Default | Description
--------- | ------- | -----------
`x` | NA | the predictor data x
`y` | NA | the prediction data y (truth)
`task`| NA | One of the following strings: 'binary', 'multi_class', 'multi_label', or 'continuous'.
`model_id` | None | the model_id to be used
`folds` | 5 | number of held-out evaluation subsets
`shuffle` | True | if data is shuffled before splitting
`average` | None (task default) | 'binary', 'micro', 'macro', 'samples', or 'weighted'
`metric` | required | the metric against which the validation is performed
`asc` | None | should be True if metric is a loss
`saved` | bool | if a model saved on local machine should be used
`custom_objects` | dict | if the model has a custom object, pass it here
`model_factory` | callable or None | Reconstructs a saved Torch model when its architecture cannot be recovered from the archive.

The above arguments are for the <code>evaluate</code> attribute of the <code>Evaluate</code> object.
