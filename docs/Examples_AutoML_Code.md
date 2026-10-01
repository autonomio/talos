[BACK](Examples_AutoML.md)

# AutoML

```python
import talos
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_iris(return_X_y=True)
x, y = x[y < 2].astype('float32'), y[y < 2]
x, x_test, y, y_test = train_test_split(x, y, test_size=.1, stratify=y, random_state=17)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, stratify=y, random_state=17)
scaler = StandardScaler().fit(x_train)
x_train, x_val, x_test = [scaler.transform(part) for part in (x_train, x_val, x_test)]

autom8 = talos.autom8.AutoScan(task='binary', experiment_name='iris_automl', max_param_values=2)
# AutoParams supplies all required keys; bound this educational run explicitly.
p = talos.autom8.AutoParams(task='binary', network=False, resample_params=1).params
p.update({'epochs': [2], 'first_neuron': [8], 'hidden_layers': [0],
          'batch_size': [16], 'dropout': [0.], 'losses': ['binary_crossentropy'],
          'activation': ['relu', 'elu']})

scan_object = autom8.start(x=x_train, y=y_train, x_val=x_val, y_val=y_val,
                           params=p, round_limit=2, seed=17, backend='tensorflow')
assert len(scan_object.data) == 2
```
