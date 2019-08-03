[BACK](Examples_AutoML.md)

# AutoML

```python

x, y = talos.templates.datasets.cervical_cancer()

# we spare 10% of data for testing later
x, y, x_test, y_test = wrangle.array_split(x, y, .1)

# then validation split
x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .2)

autom8 = talos.autom8.AutoScan('binary', 5)

autom8.start(x=x_train,
             y=y_train,
             x_val=x_val,
             y_val=y_val,
             fraction_limit=0.000001)
```
