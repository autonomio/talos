## LR Normalizer

As one experiment may include more than one optimizer, and optimizers generally have default learning rates in different order of magnitudes, lr_normalizer can be used to allow simultanously including different optimizers and different degrees of learning rates into the Talos experiment.

```python
from talos.model.normalizers import lr_normalizer
# model first part is here ...
model.compile(loss='binary_crossentropy',
              optimizer=params['optimizer'](lr_normalizer(params['lr'], params['optimizer'])),
              metrics=['accuracy'])
# model ending part is here ...
```

<aside class="notice">
The lr_normalizer needs to be invoked explicitly
</aside>
