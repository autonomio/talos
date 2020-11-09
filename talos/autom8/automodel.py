class AutoModel:

    def __init__(self, task, experiment_name, metric=None):

        '''

        Creates an input model for Scan(). Optimized for being used together
        with Params(). For example:

        p = talos.AutoParams().params
        model = talos.AutoModel(task='binary').model

        talos.Scan(x, y, p, model)

        NOTE: the parameter space from Params() is very large, so use limits
        in or reducers in Scan() accordingly.

        task : string or None
            If 'continuous' then mae is used for metric, if 'binary',
            'multiclass', or 'multilabel', f1score is used. Accuracy is always
            used.
        experiment_name | str | Must be same as in `Scan()`
        metric : None or list
            You can also input a list with one or more custom metrics or names
            of Keras or Talos metrics.
        '''

        from talos.utils.experiment_log_callback import ExperimentLogCallback

        self.task = task
        self.experiment_name = experiment_name
        self.metric = metric

        if self.task is not None:
            self.metrics = self._set_metric()
        elif self.metric is not None and isinstance(self.metric, list):
            self.metrics = self.metric + ['acc']
        else:
            print("Either pick task or provide list as input for metric.")

        # create the model
        self.model = self._create_input_model
        self.callback = ExperimentLogCallback

    def _set_metric(self):

        """Sets the metric for the model based on the experiment type
        or a list of metrics from user."""

        import talos as ta

        if self.task in ['binary', 'multiclass', 'multilabel']:
            return [ta.utils.metrics.f1score, 'acc']
        elif self.task == 'continuous':
            return [ta.utils.metrics.mae, 'acc']

    def _create_input_model(self, x_train, y_train, x_val, y_val, params):

        import wrangle as wr

        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dropout, Flatten
        from tensorflow.keras.layers import LSTM, Conv1D, SimpleRNN, Dense, Bidirectional

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
                            activation='relu',
                            kernel_initializer=params['kernel_initializer']))

        model.add(Dropout(params['dropout']))

        # add hidden layers to the model
        from talos.model.hidden_layers import hidden_layers
        hidden_layers(model, params, 1)

        # get the right activation and last_neuron based on task
        from talos.model.output_layer import output_layer
        activation, last_neuron = output_layer(self.task,
                                               params['last_activation'],
                                               y_train,
                                               y_val)

        model.add(Dense(last_neuron,
                        activation=activation,
                        kernel_initializer=params['kernel_initializer']))

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
                        callbacks=[self.callback(self.experiment_name, params)],
                        validation_data=[x_val, y_val])

        # pass the output to Talos
        return out, model
