import numpy as np
from keras.optimizers import Adam, Nadam, Adadelta, SGD


loss = {'binary': ['binary_crossentropy', 'logcosh'],
        'multi_class': ['sparse_categorical_crossentropy'],
        'multi_label': ['categorical_crossentropy'],
        'continuous': ['mae']}

last_activation = {'binary': ['sigmoid'],
                   'multi_class': ['softmax'],
                   'multi_label': ['softmax'],
                   'continuous': [None]}


class Params:

    def __init__(self,
                 params=None,
                 task='binary',
                 replace=True,
                 auto=True,
                 network=True):

        '''A facility for generating or appending params dictionary.

        params : dict or None
        task : str
             'binary', 'multi_class', 'multi_label', or 'continuous'
        replace : bool
             Replace current dictionary entries with new ones.
        auto : bool
             Automatically generate or append params dictionary with
             all available parameters.
        network : bool
             Adds several network architectures as parameters. This is to be
             used as an input together with KerasModel(). If False then only
             'dense' will be added.
        '''

        self.task = task
        self.replace = replace
        self.network = network

        if params is None:
            self.params = {}
        else:
            self.params = params

        if auto:
            self.automated()

    def automated(self, shapes='fixed'):

        '''Automatically generate a comprehensive
        parameter dict to be used in Scan()

        shapes : string
            Either 'fixed' or 'sloped'

        '''

        if shapes == 'fixed':
            self.shapes()
        else:
            self.shapes_slope()
        self.layers()
        self.dropout()
        self.optimizers()
        self.activations()
        self.neurons()
        self.losses()
        self.batch_size()
        self.epochs()
        self.kernel_initializers()
        self.lr()
        if self.network:
            self.networks()
        else:
            self.params['network'] = 'dense'
        self.last_activations()

    def shapes(self):

        '''Uses triangle, funnel, and brick shapes.'''

        self._append_params('shapes', ['triangle', 'funnel', 'brick'])

    def shapes_slope(self):

        '''Uses a single decimal float for values below 0.5 to
        reduce the width of the following layer.'''

        self._append_params('shapes', np.arange(0, .6, 0.1).tolist())

    def layers(self, max_layers=6):

        self._append_params('hidden_layers', list(range(max_layers)))

    def dropout(self):

        '''Dropout from 0.0 to 0.75'''

        self._append_params('dropout', np.round(np.arange(0, .85, 0.1), 2).tolist())

    def optimizers(self, task='binary'):

        '''Adam, Nadam, SGD, and adadelta.'''
        self._append_params('optimizer', [Adam, Nadam, Adadelta, SGD])

    def activations(self):

        self._append_params('activation', ['relu', 'elu'])

    def losses(self):

        self._append_params('losses', loss[self.task])

    def neurons(self, bottom_value=8, max_value=None, steps=None):

        '''max_value and steps has to be either None or
        integer value at the same time.'''

        if max_value is None and steps is None:
            values = [int(np.exp2(i)) for i in range(3, 11)]
        else:
            values = range(bottom_value, max_value, steps)

        self._append_params('first_neuron', values)

    def batch_size(self, bottom_value=8, max_value=None, steps=None):

        '''max_value and steps has to be either None or
        integer value at the same time.'''

        if max_value is None and steps is None:
            values = [int(np.exp2(i/2)) for i in range(3, 15)]
        else:
            values = range(bottom_value, max_value, steps)

        self._append_params('batch_size', values)

    def epochs(self, bottom_value=50, max_value=None, steps=None):

        '''max_value and steps has to be either None or
        integer value at the same time.'''

        if max_value is None and steps is None:
            values = [int(np.exp2(i/2))+50 for i in range(3, 15)]
        else:
            values = range(bottom_value, max_value, steps)

        self._append_params('epochs', values)

    def kernel_initializers(self):

        self._append_params('kernel_initializer',
                            ['glorot_uniform', 'glorot_normal',
                             'random_uniform', 'random_normal'])

    def lr(self):

        a = np.round(np.arange(0.01, 0.2, 0.02), 3).tolist()
        b = np.round(np.arange(0, 1, 0.2), 2).tolist()
        c = list(range(0, 11))

        self._append_params('lr', a + b + c)

    def networks(self):

        '''Adds four different network architectures are parameters:
        dense, simplernn, lstm, conv1d.'''

        self._append_params('network', ['dense',
                                        'simplernn',
                                        'lstm',
                                        'bidirectional_lstm',
                                        'conv1d'])

    def last_activations(self):

        self._append_params('last_activation', last_activation[self.task])

    def _append_params(self, label, values):

        if self.replace is False:
            try:
                self.params[label]
            except KeyError:
                self.params[label] = values

        else:
            self.params[label] = values
