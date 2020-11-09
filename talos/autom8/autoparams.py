import numpy as np
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta, SGD


loss = {'binary': ['binary_crossentropy', 'logcosh'],
        'multi_class': ['sparse_categorical_crossentropy'],
        'multi_label': ['categorical_crossentropy'],
        'continuous': ['mae']}

last_activation = {'binary': ['sigmoid'],
                   'multi_class': ['softmax'],
                   'multi_label': ['softmax'],
                   'continuous': [None]}


class AutoParams:

    def __init__(self,
                 params=None,
                 task='binary',
                 replace=True,
                 auto=True,
                 network=True,
                 resample_params=4):

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
        resample_params | int or False | The number of values per parameter
        '''

        self._task = task
        self._replace = replace
        self._network = network

        if params is None:
            self.params = {}
        else:
            self.params = params
        if auto:
            self._automated()

        if resample_params is not False:
            self.resample_params(resample_params)

    def _automated(self, shapes='fixed'):

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
        if self._network:
            self.networks()
        else:
            self.params['network'] = ['dense']
        self.last_activations()

    def shapes(self, shapes='auto'):

        '''Uses triangle, funnel, and brick shapes.'''

        if shapes == 'auto':
            self._append_params('shapes', ['triangle', 'funnel', 'brick'])
        else:
            self._append_params('shapes', shapes)

    def shapes_slope(self, min_slope=0, max_slope=.6, steps=.1):

        '''Uses a single decimal float for values below 0.5 to
        reduce the width of the following layer.'''

        self._append_params('shapes', np.arange(min_slope,
                                                max_slope,
                                                steps).tolist())

    def layers(self, min_layers=0, max_layers=6, steps=1):

        self._append_params('hidden_layers',
                            list(range(min_layers, max_layers, steps)))

    def dropout(self, min_dropout=0, max_dropout=.85, steps=0.1):

        self._append_params('dropout',
                            np.round(np.arange(min_dropout,
                                               max_dropout,
                                               steps), 2).tolist())

    def optimizers(self, optimizers='auto'):

        '''If `optimizers='auto'` then optimizers will be picked based on
        automatically. Otherwise input a list with one or
        more optimizers will be used.
        '''

        if optimizers == 'auto':
            self._append_params('optimizer', [Adam, Nadam, Adadelta, SGD])
        else:
            self._append_params('optimizer', optimizers)

    def activations(self, activations='auto'):

        '''If `activations='auto'` then activations will be picked based on
        automatically. Otherwise input a list with one or
        more activations will be used.
        '''

        if activations == 'auto':
            activations = ['relu', 'elu']

        self._append_params('activation', activations)

    def losses(self, losses='auto'):

        '''If `losses='auto'` then losses will be picked based on
        `AutoParam()` argument `task`. Otherwise input a list with one or
        more losses will be used.
        '''

        if losses == 'auto':
            self._append_params('losses', loss[self._task])
        else:
            self._append_params('losses', losses)

    def neurons(self, min_neuron=8, max_neuron=None, steps=None):

        '''`max` and `steps` has to be either `None` or
        integer value at the same time.'''

        if max_neuron is None and steps is None:
            values = [int(np.exp2(i)) for i in range(3, 11)]
        else:
            values = list(range(min_neuron, max_neuron, steps))

        self._append_params('first_neuron', values)

    def batch_size(self, min_size=8, max_size=None, steps=None):

        '''`max_size` and `steps` has to be either `None` or
        integer value at the same time.'''

        if max_size is None and steps is None:
            values = [int(np.exp2(i/2)) for i in range(3, 15)]
        else:
            values = list(range(min_size, max_size, steps))

        self._append_params('batch_size', values)

    def epochs(self, min_epochs=50, max_epochs=None, steps=None):

        '''`max_epochs` and `steps` has to be either `None` or
        integer value at the same time.'''

        if max_epochs is None and steps is None:
            values = [int(np.exp2(i/2))+50 for i in range(3, 15)]
        else:
            values = list(range(min_epochs, max_epochs, steps))

        self._append_params('epochs', values)

    def kernel_initializers(self, kernel_inits='auto'):

        '''
        kernel_inits | list | one or more kernel initializers
        '''

        if kernel_inits == 'auto':
            self._append_params('kernel_initializer',
                                ['uniform', 'normal', 'he_normal',
                                 'he_uniform', 'lecun_normal',
                                 'glorot_uniform', 'glorot_normal',
                                 'random_uniform', 'random_normal'])
        else:
            self._append_params('kernel_initializer', kernel_inits)

    def lr(self, learning_rates='auto'):

        '''If `learning_rates='auto'` then a very wide range of learning rates
        will be added. Otherwise a list with one or more learning rates
        is used.

        NOTE: talos.utils.lr_normalizer should be used if more than one optimizer
        is used in the experiment
        '''

        if learning_rates == 'auto':

            a = np.round(np.arange(0.01, 0.2, 0.02), 3).tolist()
            b = np.round(np.arange(0, 1, 0.2), 2).tolist()
            c = list(range(0, 11))

            self._append_params('lr', a + b + c)

        else:

            self._append_params('lr', learning_rates)

    def networks(self, networks='auto'):

        '''If `network='auto'` then dense, simplernn, lstm, conv1d, and
        bidirectional_lstm are added. Otherwise a list with one or more
        network architectures is used.
        '''

        if networks == 'auto':
            self._append_params('network', ['dense',
                                            'simplernn',
                                            'lstm',
                                            'bidirectional_lstm',
                                            'conv1d'])
        else:
            self._append_params('network', networks)

    def last_activations(self, last_activations='auto'):

        '''If `last_activations='auto'` then activations will be picked
        automatically based on `AutoParams` property `task`.
        Otherwise input a list with one or more activations will be used.
        '''

        if last_activations == 'auto':
            self._append_params('last_activation', last_activation[self._task])
        else:
            self._append_params('last_activation', last_activations)

    def resample_params(self, n):

        '''Resamples params dictionary so that `n` values are present for each
        parameter.'''

        from wrangle import dic_resample_values

        self.params = dic_resample_values(self.params, n)

    def _append_params(self, label, values):

        if self._replace is False:
            try:
                self.params[label]
            except KeyError:
                self.params[label] = values

        else:
            self.params[label] = values
