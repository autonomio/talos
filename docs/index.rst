=============================
Talos User Manual
=============================

.. image:: https://travis-ci.org/autonomio/talos.svg?branch=master
    :target: https://travis-ci.org/autonomio/talos

.. image:: https://coveralls.io/repos/github/autonomio/talos/badge.svg?branch=master
    :target: https://coveralls.io/github/autonomio/talos?branch=master


.. image:: https://gemnasium.com/badges/github.com/autonomio/talos.svg
    :target: https://gemnasium.com/github.com/autonomio/talos


This document covers the functionality of Talos, a hyperparameter optimizer for Keras models. If you're looking for a high level overview of the capabilities, you might find [Talos overview]_ more useful. 


Installation
------------

Installing from package::

    pip install talos

Installing from git::

    pip install git+https://github.com/autonomio/talos.git


Usage
-----

There are three components that are involved in setting up an experiment (a scan of given parameter space). 

1) a Keras model 

2) a Python dictionary 

3) talos.Scan() command 

In addition some imports are needed from Keras, and from talos. 


Usage Notes
-----------

- Models need to have a model.fit() object and model in the return statement

- The model needs to be inside a function (which is passed to the talos.Scan()

- You must use the exact names that are used by Keras for each parameter

- If lr_normalizer is not used with optimizer, then use::

    model.compile(optimizer=params['optimizer']())
    
- If you want to have a given parameter in the result dataframe / csv log, it has to be in the param dictionary
    



Parameter Dictionary
--------------------

The parameter dictionary is a simple python dictionary with keys and values. The values are accepted as lists of values, as tuples that represent ranges, and as single values inside a list. 

+-------------------+-------------------------+-------------------------+
|                   |                         |                         |
| INPUT TYPE        | REQUIRED INPUT          | FORMAT                  |
+===================+=========================+=========================+
| List              | one or more values      | depends on the parameter|
+-------------------+-------------------------+-------------------------+
| Tuple             | start, stop, steps      | int or float            |
+-------------------+-------------------------+-------------------------+

None is accepted as value for any parameter. In summary: 

- Inputting just one parameter is enough
- Any Keras parameter can be included 
- Any number of parameters can be included 
- Any number of variations can be included 

A simple example of a parameter dictionary::

      p = {'lr': (0.5, 5, 10),
           'first_neuron':[4, 8, 16, 32, 64],
           'hidden_layers':[0, 1, 2, 3, 4],
           'batch_size': (2, 30, 10),
           'epochs': [300],
           'dropout': (0, 0.5, 5),
           'weight_regulizer':[None],
           'emb_output_dims': [None],
           'optimizer': [Adam, Nadam],
           'losses': [logcosh, binary_crossentropy],
           'activation':[relu, elu],
           'last_activation': [softmax]}

Reduction Methods
-----------------
Because the parameter space expands according to n! factorial, in a typical scenario the number of possible permutations becomes very large. Talos provides two approaches to reducing the number of permutations: 

- random reduction 
- non-random reduction

The random approaches cut down the possible permutations before starting the scan, whereas non-random approaches cut down the permuations on-going based on results. For example, if batch sizes in the upper bound of the set parameter boundary perform poorly, they will be automatically reduced. Talos provides several ways to effect this process. These are covered in detail in the below section titled 'Input Parameters'.

Input Parameters
----------------
NOTE: This section should not be confused with the section covering the Keras model parameters that are being scanned/optimized. Input parameters refer to the parameters required and accepted within Talos commands. 


+-------------------+-------------------------+----------------------------------+
|                   |                         |                                  |
| INPUT PARAMETER   | REQUIRED INPUT          | DESCRIPTION                      |
+===================+=========================+==================================+
| X                 | 1d or 2d array          | predictor variables              |
+-------------------+-------------------------+----------------------------------+
| Y                 | 1d or 2d array          | prediction variable              |
+-------------------+-------------------------+----------------------------------+
| params            | python dict             | parameter boundaries             |
+-------------------+-------------------------+----------------------------------+
| dataset_name      | string                  | name of the dataset              |
+-------------------+-------------------------+----------------------------------+
| experiment_no     | string                  | number of experiment             |
+-------------------+-------------------------+----------------------------------+
| model             | a function              | any Keras model                  |
+-------------------+-------------------------+----------------------------------+
| val_split         | float                   | e.g. .3 for 30% to val           |
+-------------------+-------------------------+----------------------------------+
| shuffle           | True or False           | shuffle before split             |
+-------------------+-------------------------+----------------------------------+
| search_method     | string label            | 'random', 'linear', 'random'     |
+-------------------+-------------------------+----------------------------------+
| save_best_model   | True or False           | best model will be saved         |
+-------------------+-------------------------+----------------------------------+
| reduction_method  | string label            | 'spear' or 'random'              |
+-------------------+-------------------------+----------------------------------+
| reduction_interval| int                     | reduce after every n permutations|
+-------------------+-------------------------+----------------------------------+
| reduction_metric  | string label            | the metric used for evaluation   |
+-------------------+-------------------------+----------------------------------+
| grid_downsample   | float                   | e.g. 0.01 for 1%                 |
+-------------------+-------------------------+----------------------------------+
| talos_log_name    | string                  | name of the master log           |
+-------------------+-------------------------+----------------------------------+
| debug             | True or False           | invokes debug logging            |
+-------------------+-------------------------+----------------------------------+

NOTE: None of these have effect on the actual parameter permutations. Those are set within the parameter dictionary explained in the above section. 

x
.
This needs be a 1d or 2d array with predictor features.

y
.
This needs to be a 1d or 2d array with prediction values.

params
......
This is the parameter dictionary explained in the above sections. Note that the any Keras parameter can be added simply by adding it to the dictionary and the referencing it in the Keras model with the dictionary key. 


val_split
.........

The validation split that will be used for the experiment. By default .3 i.e. 30% goes to validation dataset. 

shuffle
.......

Defines if the dataset should be shuffled before validation split is performed. By default True. Note that time series data should never be shuffled. 

search_method
.............

Three modes are offered: 'random', 'linear', and 'reverse'. Random picks randomly one permutation and then removes it from the search grid. Linear starts from the beginning of the grid, and reverse from the end.

reduction_method
................

There is currently one reduction algorithm available 'spear'. It is based on an approach where depending on the 'reduction_interval' and 'reduction_window' poorly performing parameters are dropped from the scan. If you would like to see a specific algorithm implemented, please create an issue for it.

reduction_interval
..................

The number of rounds / permutation attempts after which the reduction method will be applied. The 'reduction_method' must be set to other than None for this to take effect.

reduction_window
................

The number of rounds / permutation attempts for looking back when applying the reduction_method. For continuous optimization, this should be less than reduction_interval or the same.

grid_downsampling
.................

Takes in a float value based on which a fraction of the total parameter grid will be picked randomly.

early_stopping
..............

Provides a callback functionality where once val_loss (validation loss) is no longer dropping, based on the setting, the round will be terminated. Results for the round will be still recorded before moving on to the next permutation. Accepts a string values 'moderate' and 'strict', or a list with two int values (min_delta, patience). Where min_delta indicates the threshhold for change where the round will be flagged for termination (e.g. 0 means that val_loss is not changing) and patience indicates the number of epochs counting from the flag being raised before the round is actual terminated.

dataset_name
............

This information is used for the master log and naming the experiment results round results .csv file.

experiment_no
.............

This will be appended to the round results .csv file and together with the dataset_name form a unique handler for the experiment.  

talos_log_name
..............

The path to the master log file where a log entry is created for every single scan event together with meta-information such as what type of prediction challenge it is, how the data is transformed (e.g. one-hot encoded). This data can be useful for training models for the purpose of optimizing models. That's right, models that make models.

By default talos.log is in the present working directory. It's better to change this to something where it has persistence.

debug
.....

Useful when you don't want records to be made in to the master log (./talos.log)

Links
-----

.. [Talos_Overview] https://github.com/autonomio/talos/blob/master/README.md
