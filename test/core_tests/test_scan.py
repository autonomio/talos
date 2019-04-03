#!/usr/bin/env python

from __future__ import print_function

from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.optimizers import Adamax, RMSprop, Nadam
from keras.activations import relu, sigmoid

from sklearn.model_selection import train_test_split as splt

from talos.scan.Scan import Scan
from talos.commands.reporting import Reporting

import talos as ta


# single values
def values_single_params():
    return {'lr': [1],
            'first_neuron': [4],
            'hidden_layers': [2],
            'batch_size': [100],
            'epochs': [2],
            'dropout': [0],
            'shapes': ['brick'],
            'optimizer': [Adam],
            'losses': [binary_crossentropy,
                       sparse_categorical_crossentropy,
                       categorical_crossentropy,
                       mean_squared_error],
            'activation': ['relu'],
            'last_activation': ['softmax']}


# lists of values
def values_list_params():
    return {'lr': [1, 2],
            'first_neuron': [4, 4],
            'hidden_layers': [2, 2],
            'batch_size': [100, 200],
            'epochs': [1, 2],
            'dropout': [0, 0.1],
            'shapes': ['brick', 'funnel', 'triangle', 0.2],
            'optimizer': [Adam, Adagrad, Adamax, RMSprop, Adadelta, Nadam, SGD],
            'losses': ['binary_crossentropy',
                       'sparse_categorical_crossentropy',
                       'categorical_crossentropy',
                       'mean_squared_error'],
            'activation': ['relu', 'elu'],
            'last_activation': ['softmax']}


# range of values
def values_range_params():
    return {'lr': (0.5, 5, 10),
            'first_neuron': (4, 100, 5),
            'hidden_layers': (0, 5, 5),
            'batch_size': (200, 300, 10),
            'epochs': (1, 5, 4),
            'dropout': (0, 0.5, 5),
            'shapes': ['funnel'],
            'optimizer': [Nadam],
            'losses': [binary_crossentropy,
                       sparse_categorical_crossentropy,
                       categorical_crossentropy,
                       mean_squared_error],
            'activation': [relu],
            'last_activation': [sigmoid]}


"""
The tests below have to serve several purpose:

- test possible input methods to params dict
- test binary, multi class, multi label and continuous problems
- test all Scan arguments

Each problem type is presented as a Class, and contains three
experiments using single, list, or range inputs. There is an
effort to test as many scenarios as possible here, so be
inventive / experiment! Doing well with this part of the testing,
there is a healthy base for a more serious approach to ensuring
procedural integrity.

"""


def get_params(task):

        """

        Helper that allows the tests to feed from same
        params dictionaries.

        USE: values_single, values_list, values_range = get_appropriate_loss(0)

        0 = binary
        1 = 1d multi class
        2 = 2d multi label
        3 = continuous / regression

        """

        # first create the params dict
        values_single = values_single_params()
        values_list = values_list_params()
        values_range = values_range_params()

        # then limit the losses according to prediction task
        values_single['losses'] = [values_single_params()['losses'][task]]
        values_list['losses'] = [values_list_params()['losses'][task]]
        values_range['losses'] = [values_range_params()['losses'][task]]

        return values_single, values_list, values_range


class BinaryTest:

    def __init__(self):

        # read the params dictionary with the right loss
        self.values_single, self.values_list, self.values_range = get_params(0)

        # prepare the data for the experiment
        self.x, self.y = ta.templates.datasets.cervical_cancer()
        self.x = self.x[:300]
        self.y = self.y[:300]
        self.model = ta.templates.models.cervical_cancer

        # split validation data
        self.x_train, self.x_val, self.y_train, self.y_val = splt(self.x,
                                                                  self.y,
                                                                  test_size=0.2)

    def values_single_test(self):
        print("BinaryTest : Running values_single_test...")

        Scan(self.x,
             self.y,
             params=self.values_single,
             model=ta.templates.models.cervical_cancer)

    def values_list_test(self):
        print("BinaryTest : Running values_list_test...")
        Scan(self.x_train,
             self.y_train,
             x_val=self.x_val,
             y_val=self.y_val,
             params=self.values_list,
             round_limit=5,
             dataset_name='BinaryTest',
             experiment_no='000',
             model=ta.templates.models.cervical_cancer,
             random_method='crypto_uniform',
             seed=2423,
             search_method='linear',
             reduction_method='correlation',
             reduction_interval=2,
             reduction_window=2,
             reduction_threshold=0.2,
             reduction_metric='val_loss',
             reduce_loss=True,
             last_epoch_value=True,
             clear_tf_session=False,
             disable_progress_bar=True,
             debug=True)

    # comprehensive
    def values_range_test(self):
        print("BinaryTest : Running values_range_test...")
        Scan(self.x_train,
             self.y_train,
             params=self.values_range,
             model=ta.templates.models.cervical_cancer,
             grid_downsample=0.0001,
             permutation_filter=lambda p: p['first_neuron'] * p['hidden_layers'] < 220,
             random_method='sobol',
             reduction_method='correlation',
             reduction_interval=2,
             reduction_window=2,
             reduction_threshold=0.2,
             reduction_metric='val_acc',
             reduce_loss=False,
             debug=True)


class MultiLabelTest:

    def __init__(self):

        # read the params dictionary with the right loss
        self.values_single, self.values_list, self.values_range = get_params(2)

        self.x, self.y = ta.templates.datasets.iris()
        self.x_train, self.x_val, self.y_train, self.y_val = splt(self.x,
                                                                  self.y,
                                                                  test_size=0.2)

    def values_single_test(self):
        print("MultiLabelTest : Running values_single_test...")
        Scan(self.x,
             self.y,
             params=self.values_single,
             model=ta.templates.models.iris)

    def values_list_test(self):
        print("MultiLabelTest : Running values_list_test...")
        Scan(self.x,
             self.y,
             x_val=self.x_val,
             y_val=self.y_val,
             params=self.values_list,
             round_limit=5,
             dataset_name='MultiLabelTest',
             experiment_no='000',
             model=ta.templates.models.iris,
             random_method='crypto_uniform',
             seed=2423,
             search_method='linear',
             permutation_filter=lambda p: p['first_neuron'] * p['hidden_layers'] < 9,
             reduction_method='correlation',
             reduction_interval=2,
             reduction_window=2,
             reduction_threshold=0.2,
             reduction_metric='val_loss',
             reduce_loss=True,
             last_epoch_value=True,
             clear_tf_session=False,
             disable_progress_bar=True,
             debug=True)

    # comprehensive
    def values_range_test(self):
        print("MultiLabelTest : Running values_range_test...")
        Scan(self.x,
             self.y,
             params=self.values_range,
             model=ta.templates.models.iris,
             grid_downsample=0.0001,
             random_method='sobol',
             reduction_method='correlation',
             reduction_interval=2,
             reduction_window=2,
             reduction_threshold=0.2,
             reduction_metric='val_acc',
             reduce_loss=False,
             debug=True)


class ReportingTest:

    def __init__(self):

        print("ReportingTest : Running Binary test...")

        r = Reporting('BinaryTest_000.csv')

        x = r.data
        x = r.correlate()
        x = r.high()
        x = r.low()
        x = r.rounds()
        x = r.rounds2high()
        x = r.best_params()
        x = r.plot_corr()
        x = r.plot_hist()
        x = r.plot_line()

        print("ReportingTest : Running MultiLabel test...")
        r = Reporting('MultiLabelTest_000.csv')

        x = r.data
        x = r.correlate()
        x = r.high()
        x = r.low()
        x = r.rounds()
        x = r.rounds2high()
        x = r.best_params()
        x = r.plot_corr()
        x = r.plot_hist()
        x = r.plot_line()

        del x


class DatasetTest:

    def __init__(self):

        print("DatasetTest : Running tests...")
        x = ta.templates.datasets.icu_mortality()
        x = ta.templates.datasets.icu_mortality(100)
        x = ta.templates.datasets.titanic()
        x = ta.templates.datasets.iris()
        x = ta.templates.datasets.cervical_cancer()
        x = ta.templates.datasets.breast_cancer()
        x = ta.templates.params.iris()
        x = ta.templates.params.breast_cancer()
