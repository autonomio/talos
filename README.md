<img alt='Hyperparameter scanner for Keras Models' src='https://raw.githubusercontent.com/autonomio/talos/master/logo.png' width=250px>

## Hyperparameter Scanning and Optimization for Keras  [![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=Hyperparameter%20optimization%20for%20humans&url=https://github.com/autonomio/talos&hashtags=AI,deeplearning,keras)

![Travis branch](https://img.shields.io/travis/autonomio/talos/master.svg)[![Coverage Status](https://coveralls.io/repos/github/autonomio/talos/badge.svg?branch=master)](https://coveralls.io/github/autonomio/talos?branch=master)

Talos is a solution that helps finding hyperparameter configurations for Keras models. To perform hyperparameter optimization with Talos, there is no need to learn any new syntax, or change anything in the way Keras models are created. Keras functionality is fully exposed, and any parameter can be included in the scans.

Talos is made for data scientists and data engineers that want to remain in complete control of their Keras models, but are tired of mindless parameter hopping and confusing optimization solutions that add complexity instead of taking it away.

See the example Notebook [HERE](https://github.com/autonomio/talos/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb)

Read the User Manual [HERE](https://github.com/autonomio/talos/blob/master/docs/index.rst)

Read a Talos field-report on [Hyperparameter Optimization](https://towardsdatascience.com/hyperparameter-optimization-with-keras-b82e6364ca53)

## Development Objectives

Currently Talos yields state-of-the-art results (e.g. Iris dataset 100% and Wisconsin Breast Cancer dataset 99.4%) across a range of prediction tasks in a semi-automatic manner, while providing the simplest available method for hyperparameter optimization with Keras.

Read the roadmap [HERE](https://github.com/autonomio/talos/blob/master/docs/roadmap.rst)

## Benefits

Based on a review of more than 30 hyperparameter optimization and scanning solutions, Talos offers the most intuitive, easy-to-learn, and permissive access to important hyperparameter optimization capabilities.

- works with ANY Keras model
- very easy to implement
- adds zero new overhead
- provides several ways to reduce random-search complexity
- no need to learn any new syntax
- no blackbox / other statistical complexity
- improved f1 performance metric for binary, multi-label, multi-class and continuous predictions

## Install

    pip install talos

Or from git repo:

    pip install git+https://github.com/autonomio/talos.git

## How to use

Let's consider an example of a simple Keras model:

    model = Sequential()
    model.add(Dense(8, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(optimizer='adam',
                  loss=categorical_crossentropy,
                  metrics=['acc'])

    out = model.fit(x_train, y_train,
                    batch_size=20,
                    epochs=200,
                    verbose=0,
                    validation_data=[x_val, y_val])

To prepare the model for a talos scan, we simply replace the parameters we want to include in the scans with references to our parameter dictionary (example of dictionary provided below). The below example code complete [here](https://github.com/autonomio/talos/blob/master/examples/iris.py).

	def iris_model(x_train, y_train, x_val, y_val, params):

	    model = Sequential()
	    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1], activation=params['activation']))
	    model.add(Dropout(params['dropout']))
	    model.add(Dense(y_train.shape[1], activation=params['last_activation']))

	    model.compile(optimizer=params['optimizer'],
	                  loss=params['losses'],
	                  metrics=['acc'])

	    out = model.fit(x_train, y_train,
	                    batch_size=params['batch_size'],
	                    epochs=params['epochs'],
	                    verbose=0,
	                    validation_data=[x_val, y_val])

	    return out, model

As you can see, the only thing that changed, is the values that we provide for the parameters. We then pass the parameters with a dictionary:

	p = {'lr': (2, 10, 30),
	     'first_neuron':[4, 8, 16, 32, 64, 128],
	     'hidden_layers':[2,3,4,5,6],
	     'batch_size': [2, 3, 4],
	     'epochs': [300],
	     'dropout': (0, 0.40, 10),
	     'weight_regulizer':[None],
	     'emb_output_dims': [None],
	     'optimizer': [Adam, Nadam],
	     'losses': [categorical_crossentropy, logcosh],
	     'activation':[relu, elu],
	     'last_activation': [softmax]}

The above example is a simple indication of what is possible. Any parameter that Keras accepts, can be included in the dictionary format.

Talos accepts lists with values, and tuples (start, end, n). Learning rate is normalized to 1 so that for each optimizer, lr=1 is the default Keras setting. Once this is all done, we can run the scan:

	h = ta.Scan(x, y,
              params=p,
              dataset_name='first_test',
              experiment_no='2',
              model=iris_model,
              grid_downsample=0.5)


## Not All Randomness Are Created Equal

The main optimization strategy focus in Talos is to provide the gold standard random search capabilities. Talos implements three kinds of random generation methods:

- True / Quantum randomness
- Pseudo randomness
- Quasi randomness

The currently implemented methods are:

- Quantum randomness (vacuum based)
- Ambient Sound based randomness
- Sobol sequences
- Halton sequences
- Latin hypercube
- Improved Latin hypercube
- Latin hypercube with a Sudoku-style constraint
- Uniform Mersenne
- Cryptographically sound uniform

Each method differs in discrepancy and other observable aspects.

## More on Optimization Strategies

Talos supports several common optimization strategies:

- Random search
- Grid search
- Manually assisted random or grid search
- Correlation based optimization

The object of abstraction is the keras model configuration, of which n number of permutations is tried in a  Talos experiment.

As opposed to adding more complex optimization strategies, which are widely available in various solutions, Talos focus is on:

- adding variations of random variable picking
- reducing the workload of random variable picking

As it stands, both of these approaches are currently under leveraged by other solutions, and under represented in the literature.

## Built With

* [Numpy](http://numpy.org) - Scientific Computing
* [Keras](https://keras.io/) - Deep Learning for Humans

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/autonomio/talos/blob/master/LICENSE) file for details
