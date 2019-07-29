## Evaluate()

The models that result from the experiment `Scan` object can be evaluated with `Evaluate()`. This way one or more models may be picked for deployment using k-fold cross-validation in a straightforward manner.

> Evaluating model generality

```python
from talos import Evaluate

# create the evaluate object
e = Evaluate(scan_object)

# perform the evaluation
e.evaluate(x, y, average='macro')
```

NOTE: It's very important to save part of your data for evaluation, and keep it completely separated from the data you use for the actual experiment. A good approach would be where 50% of the data is saved for evaluation.

### Evaluate Functions

See the function docstring for a more detailed description.

**`evaluate`** The highest result for a given metric

### Evaluate Arguments

Parameter | Default | Description
--------- | ------- | -----------
x | NA | the predictor data x
y | NA | the prediction data y (truth)
model_id | None | the model_id to be used
folds | None | number of folds to be used for cross-validation
shuffle | None | if data is shuffled before splitting
average | 'binary' | 'binary', 'micro', 'macro', 'samples', or 'weighted'
metric | None | the metric against which the validation is performed
asc | None | should be True if metric is a loss

<aside class='notice'> The above arguments are for the <code>evaluate</code> attribute of the <code>Evaluate</code> object.</aside>
