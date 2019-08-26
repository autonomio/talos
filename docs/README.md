# Quick start

```python
pip install talos
```
See [here](Install_Options.md) for more options.


# Minimal Example

Your Keras model **WITHOUT** Talos:

```python
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x, y)
```
Your Keras model **WITH** Talos:


```python
model = Sequential()
model.add(Dense(12, input_dim=8, activation=params['activation']))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', params['optimizer'])
model.fit(x, y)

```
See the code complete example [here](http).

# Typical Use-cases

The most common use-case of Talos is a hyperparameter scan based on an already created Keras or TensorFlow model. In addition to the [input model](), a hyperparameter scan with Talos involves `talos.Scan()` command and a [parameter dictionary]().

After completing an experiment, results can be analyzed and visualized. Once a decision have been made if a) the experiment should be reconfigured and continue or b) sufficient level of performance have already been found, model candidates can be automatically evaluated and used for prediction.

Talos also supports easy deployment of models and experiment assets from the experiment environment to production or other systems.

# System Requirements

- Linux, Mac OSX or Windows system
- Python 3.5 or higher (Talos versions 0.5.0 and below support 2.7)
- TensorFlow, Theano, or CNTK

Talos incorporates grid, random, and probabilistic hyperparameter optimization strategies, with focus on maximizing the flexibility, efficiency, and result of random strategy. Talos users benefit from access to pseudo, quasi, true, and quantum random methods.  
