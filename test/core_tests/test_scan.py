#!/usr/bin/env python

from __future__ import print_function

from keras.losses import categorical_crossentropy, logcosh
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.optimizers import Adamax, RMSprop, Nadam
from keras.activations import softmax, relu, sigmoid

import talos as ta

from talos.model.examples import iris_model, cervix_model


p1 = {'lr': [1],
      'first_neuron': [4],
      'hidden_layers': [2],
      'batch_size': [50],
      'epochs': [1],
      'dropout': [0],
      'shapes': ['stairs', 'triangle', 'hexagon', 'diamond',
                 'brick', 'long_funnel', 'rhombus', 'funnel'],
      'optimizer': [Adam],
      'losses': [categorical_crossentropy],
      'activation': [relu],
      'last_activation': [softmax],
      'weight_regulizer': [None],
      'emb_output_dims': [None]}

p2 = {'lr': [1],
      'first_neuron': [4],
      'hidden_layers': [2],
      'batch_size': [50],
      'epochs': [1],
      'dropout': [0],
      'shapes': ['stairs'],
      'optimizer': [Adam, Adagrad, Adamax, RMSprop, Adadelta, Nadam, SGD],
      'losses': [categorical_crossentropy],
      'activation': [relu],
      'last_activation': [softmax],
      'weight_regulizer': [None],
      'emb_output_dims': [None]}

p3 = {'lr': (0.5, 5, 10),
      'first_neuron': [4, 8, 16, 32, 64],
      'hidden_layers': [2, 3, 4, 5],
      'batch_size': (2, 30, 10),
      'epochs': [3],
      'dropout': (0, 0.5, 5),
      'weight_regulizer': [None],
      'shapes': ['stairs'],
      'emb_output_dims': [None],
      'optimizer': [Nadam],
      'loss': [logcosh, binary_crossentropy],
      'activation': [relu],
      'last_activation': [sigmoid]}


class TestIris:

    def __init__(self):
        self.x, self.y = ta.datasets.iris()

    def test_scan_iris_1(self):
        print("Running Iris dataset test 1...")
        ta.Scan(self.x, self.y, params=p1, dataset_name='iris_test_1',
                experiment_no='000', model=iris_model)

    def test_scan_iris_2(self):
        print("Running Iris dataset test 2...")
        ta.Scan(self.x, self.y, params=p2, dataset_name='iris_test_2',
                experiment_no='000', model=iris_model)
        ta.Reporting('iris_test_2_000.csv')


class TestCancer:

    def __init__(self):
        self.x, self.y = ta.datasets.cervical_cancer()

    def test_scan_cancer(self):
        print("Running Cervical Cancer dataset test...")
        ta.Scan(self.x, self.y, grid_downsample=0.001, params=p3,
                dataset_name='cervical_cancer_test', experiment_no='a',
                model=cervix_model,
                reduction_method='spear', reduction_interval=5)
        ta.Reporting('cervical_cancer_test_a.csv')


class TestLoadDatasets:

    def __init__(self):
        print("Testing Load Datasets...")
        x = ta.datasets.icu_mortality()
        x = ta.datasets.icu_mortality(100)
        x = ta.datasets.titanic()
        x = ta.datasets.iris()
        x = ta.datasets.cervical_cancer()
        x = ta.datasets.breast_cancer()

        x = ta.params.iris()
        x = ta.params.breast_cancer()  # noqa
