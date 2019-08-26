# Generator

Talos provides two data generator to be used with `model.fit_generator()`. You can of course use your own generator as you would use it otherwise with stand-alone Keras.

#### basic data generator

```python
talos.utils.generator(x=x, y=y, batch_size=20)

```

#### sequence generator
```python
talos.utils.SequenceGenerator(x=x, y=y, batch_size=20)

```


NOTE: There are many performance considerations that come with using data generators in Keras. If you run in to performance issues, learn more about the experiences of other Keras users online.

You can also read the Talos thread on [using fit_generator](https://github.com/autonomio/talos/issues/11).
