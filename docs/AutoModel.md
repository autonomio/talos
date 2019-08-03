# AutoModel

`AutoModel` provides a meaningful way to test several network architectures in an automated manner. Currently there are five supported architectures:

- conv1d
- lstm
- bidirectional_lstm
- simplernn
- dense

`AutoModel` creates an input model for Scan(). Optimized for being used together with `AutoParams()` and expects one or more of the above architectures to be included in params dictionary, for example:

```python

p = {...
    'networks': ['dense', 'conv1d', 'lstm']
    ...}

```

## AutoModel Arguments

Argument | Input | Description
--------- | ------- | -----------
`task` | str or None | `binary`, `multi_label`, `multi_class`, or `continuous`
`metric` | None or list | One or more Keras metric (functions) to be used in the model

Setting `task` effects which various aspects of the model and should be set according to the specific prediction task, or set to `None` in which case `metric` input is required.
