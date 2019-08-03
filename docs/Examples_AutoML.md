# AutoML

Performing an AutoML style hyperparameter search experiment with Talos could not be any easier.

The single-file code example can be found [here](Examples_AutoML_Code.md).

### Imports

```python
import talos
import wrangle
```

### Loading Data
```python
x, y = talos.templates.datasets.cervical_cancer()

# we spare 10% of data for testing later
x, y, x_test, y_test = wrangle.array_split(x, y, .1)

# then validation split
x_train, y_train, x_val, y_val = wrangle.array_split(x, y, .2)
```

`x` and `y` are expected to be either numpy arrays or lists of numpy arrays and same applies for the case where `x_train`, `y_train`, `x_val`, `y_val` is used instead.

### Defining the Model

In this case there is no need to define the model. `talos.autom8.AutoModel()` is used behind the scenes, where several model architectures fully wired for Talos are found. We simply initiate the `AutoScan()` object first:

```python
autom8 = talos.autom8.AutoScan('binary', 5)
```

### Parameter Dictionary

There is also no need to worry about the parameter dictionary. This is handled in the background with `AutoParams()`.


### Scan()

The `Scan()` itself is started through the **`start`** property of the `AutoScan()` class object.

```python
autom8.start(x=x_train,
             y=y_train,
             x_val=x_val,
             y_val=y_val,
             fraction_limit=0.000001)
```
We pass data here just like we would do it in `Scan()` normally. Also, you are free to use any of the `Scan()` arguments here to configure the experiment. Find the description for all `Scan()` arguments [here](Scan.md#scan-arguments).
