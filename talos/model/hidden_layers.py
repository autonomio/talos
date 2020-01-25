def hidden_layers(model, params, last_neuron):
    '''HIDDEN LAYER Generator

    NOTE: 'shapes', 'first_neuron', 'dropout', and 'hidden_layers' need
    to be present in the params dictionary.

    Hidden layer generation for the cases where number
    of layers is used as a variable in the optimization process.
    Handles things in a way where any number of layers can be tried
    with matching hyperparameters.'''

    # check for the params that are required for hidden_layers

    from tensorflow.keras.layers import Dense, Dropout
    from .network_shape import network_shape
    from ..utils.exceptions import TalosParamsError

    required = ['shapes', 'first_neuron', 'dropout', 'hidden_layers', 'activation']
    for param in required:
        if param not in params:
            message = "hidden_layers requires '" + param + "' in params"
            raise TalosParamsError(message)

    layer_neurons = network_shape(params, last_neuron)

    for i in range(params['hidden_layers']):
        model.add(Dense(
            layer_neurons[i],
            kernel_initializer=params.get(
                'kernel_initializer',
                'glorot_uniform'
            ),
            kernel_regularizer=params.get('kernel_regularizer'),
            bias_initializer=params.get('bias_initializer', 'zeros'),
            bias_regularizer=params.get('bias_regularizer'),
            use_bias=params.get('use_bias', True),
            activity_regularizer=params.get('activity_regularizer'),
            kernel_constraint=params.get('kernel_constraint'),
            bias_constraint=params.get('bias_constraint'),
            activation=params.get('activation')
        ))
        model.add(Dropout(params['dropout']))
