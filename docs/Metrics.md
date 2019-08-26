# Metrics

Several common and useful performance metrics are available through `talos.utils.metrics`:

- matthews
- precision
- recall
- fbeta
- f1score
- mae
- mse
- rmae
- rmse
- mape
- msle
- rmsle

You can use these metrics as you would Keras metrics:

```python
metrics=['acc', talos.utils.metrics.f1score]
```
If you would like to add new metrics to Talos, make a [feature request](https://github.com/autonomio/talos/issues/new) or create a [pull request](https://github.com/autonomio/talos/compare).
