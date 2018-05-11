<img alt='Hyperparameter scanner for Keras Models' src='https://raw.githubusercontent.com/autonomio/talos/master/logo.png' width=250px>

## Hyperparameter Scanning for Keras

![Travis branch](https://img.shields.io/travis/autonomio/talos/master.svg)[![Coverage Status](https://coveralls.io/repos/github/autonomio/hyperio/badge.svg?branch=master)](https://coveralls.io/github/autonomio/hyperio?branch=master)

Talos provides a hyperparameter scanning solution for Keras users. There is no need to learn any new syntax, or change anything in the way Keras models are operated. Keras functionality is fully exposed, and any parameter can be included in the scans.

Talos is ideal for data scientists and data engineers that want to remain in complete control of their Keras models, but are tired of mindless parameter hopping and confusing optimization solutions that add complexity instead of taking it away.

See the example Notebook [HERE](https://github.com/autonomio/hyperio/blob/master/examples/Hyperparameter%20Optimization%20with%20Keras%20for%20the%20Iris%20Prediction.ipynb)

## Development Objective

Talos development is focused on creating a an abstraction layer for Keras, that meets the criteria of "models that build models". This means that Talos is able to, in a semi-autonomous manner find highly optimal parameter configurations for conventional prediction tasks, while being able to use that same capacity optimize itself (i.e. the optimization process) using the same approach. Thus unlocking "models that build models that build models that...". Following a reductionist approach, this goal is fulfilled by systematically building the required "blocks" one by one.

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

To prepare the model for a talos scan, we simply replace the parameters we want to include in the scans with references to our parameter dictionary (example of dictionary provided below). 

	def iris_model(x_train, y_train, x_val, y_val, params):

	    model = Sequential()
	    model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1], activation=params['activation']))
	    model.add(Dropout(params['dropout']))
	    model.add(Dense(y_train.shape[1], activation=params['last_activation']))

	    model.compile(optimizer=params['optimizer']),
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

	h = ta.Scan(x, y, params=p, experiment_name='first_test', model=iris_model, grid_downsample=0.5)

## Notes on Usage 

- Models need to have a model.fit() object and model in the return statement

- The model needs to be inside a function (which is passed to the talos.Scan()

## Options

In addition to the parameter, there are several options that can be set within the Scan() call. These values will effect the actual scan, as opposed to anything that change for each permutation.

#### val_split

The validation split that will be used for the experiment. By default .3 to validation data.

#### shuffle

If the data should be shuffle before validation split is performed. By default True.

#### search_method

Three modes are offered: 'random', 'linear', and 'reverse'. Random picks randomly one permutation and then removes it from the search grid. Linear starts from the beginning of the grid, and reverse from the end.

#### reduction_method

There is currently one reduction algorithm available 'spear'. It is based on an approach where depending on the 'reduction_interval' and 'reduction_window' poorly performing parameters are dropped from the scan. If you would like to see a specific algorithm implemented, please create an issue for it.

#### reduction_interval

The number of rounds / permutation attempts after which the reduction method will be applied. The 'reduction_method' must be set to other than None for this to take effect.

#### reduction_window

The number of rounds / permutation attempts for looking back when applying the reduction_method. For continuous optimization, this should be less than reduction_interval or the same.

#### grid_downsampling

Takes in a float value based on which a fraction of the total parameter grid will be picked randomly.

#### early_stopping

Provides a callback functionality where once val_loss (validation loss) is no longer dropping, based on the setting, the round will be terminated. Results for the round will be still recorded before moving on to the next permutation. Accepts a string values 'moderate' and 'strict', or a list with two int values (min_delta, patience). Where min_delta indicates the threshhold for change where the round will be flagged for termination (e.g. 0 means that val_loss is not changing) and patience indicates the number of epochs counting from the flag being raised before the round is actual terminated.

#### dataset_name

This information is used for the master log and naming the experiment results round results .csv file.

#### experiment_no

This will be appended to the round results .csv file and together with the dataset_name form a unique handler for the experiment.  

#### talos_log_name

The path to the master log file where a log entry is created for every single scan event together with meta-information such as what type of prediction challenge it is, how the data is transformed (e.g. one-hot encoded). This data can be useful for training models for the purpose of optimizing models. That's right, models that make models.

By default talos.log is in the present working directory. It's better to change this to something where it has persistence.

#### debug

Useful when you don't want records to be made in to the master log (./talos.log)
