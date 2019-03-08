class KerasModel:

    def __init__(self, task=None):

        '''

        Creates an input model for Scan(). Optimized for being used together
        with Params(). For example:

        p = talos.Params().params
        model = talos.KerasModel(task='binary').model

        talos.Scan(x, y, p, model)

        NOTE: the parameter space from Params() is very large, so use limits
        in or reducers in Scan() accordingly.

        task : string or list
            If 'continuous' then mae is used for metric, if 'binary',
            'multiclass', or 'multilabel', f1score is used. Accuracy is always
            used. You can also input a list with one or more custom metrics or
            names of Keras or Talos metrics.
        '''

        # pick the right metrics
        self.metrics = self._set_metric(task)

        # create the model
        self.model = self._create_input_model

    def _set_metric(self, task):

        """Sets the metric for the model based on the experiment type
        or a list of metrics from user."""

        import talos as ta

        if task is None:
            return ['acc']
        elif task in ['binary', 'multiclass', 'multilabel']:
            return [ta.utils.metric.f1score, 'acc']
        elif task == 'continuous':
            return [ta.utils.metrics.mae, 'acc']
        elif isinstance(task, list):
            return task + ['acc']

    def _create_input_model(self, x_train, y_train, x_val, y_val, params):

        import numpy as np
        import wrangle as wr

        from keras.models import Sequential
        from keras.layers import Dropout, Flatten
        from keras.layers import LSTM, Conv1D, SimpleRNN, Dense, Bidirectional

        model = Sequential()

        if params['network'] != 'dense':
            x_train = wr.array_reshape_conv1d(x_train)
            x_val = wr.array_reshape_conv1d(x_val)

        if params['network'] == 'conv1d':
            model.add(Conv1D(params['first_neuron'], x_train.shape[1]))
            model.add(Flatten())

        elif params['network'] == 'lstm':
            model.add(LSTM(params['first_neuron']))

        if params['network'] == 'bidirectional_lstm':
            model.add(Bidirectional(LSTM(params['first_neuron'])))

        elif params['network'] == 'simplernn':
            model.add(SimpleRNN(params['first_neuron']))

        elif params['network'] == 'dense':
            model.add(Dense(params['first_neuron'],
                            input_dim=x_train.shape[1],
                            activation='relu'))

        model.add(Dropout(params['dropout']))

        # add hidden layers to the model
        from talos.model.layers import hidden_layers
        hidden_layers(model, params, 1)

        # output layer (this is scetchy)
        try:
            last_neuron = y_train.shape[1]
        except IndexError:
            if len(np.unique(y_train)) == 2:
                last_neuron = 1
            else:
                last_neuron = len(np.unique(y_train))

        model.add(Dense(last_neuron,
                        activation=params['last_activation']))

        # bundle the optimizer with learning rate changes
        from talos.model.normalizers import lr_normalizer
        optimizer = params['optimizer'](lr=lr_normalizer(params['lr'],
                                                         params['optimizer']))

        # compile the model
        model.compile(optimizer=optimizer,
                      loss=params['losses'],
                      metrics=self.metrics)

        # fit the model
        out = model.fit(x_train, y_train,
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0,
                        validation_data=[x_val, y_val])

        # pass the output to Talos
        return out, model
