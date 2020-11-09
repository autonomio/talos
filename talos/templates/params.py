def titanic():

    # here use a standard 2d dictionary for inputting the param boundaries
    p = {'lr': (0.5, 5, 10),
         'first_neuron': [4, 8, 16],
         'batch_size': [20, 30, 40],
         'dropout': (0, 0.5, 5),
         'optimizer': ['Adam', 'Nadam'],
         'losses': ['logcosh', 'binary_crossentropy'],
         'activation': ['relu', 'elu'],
         'last_activation': ['sigmoid']}

    return p


def iris():

    from tensorflow.keras.optimizers import Adam, Nadam
    from tensorflow.keras.losses import logcosh, categorical_crossentropy
    from tensorflow.keras.activations import relu, elu, softmax

    # here use a standard 2d dictionary for inputting the param boundaries
    p = {'lr': (0.5, 5, 10),
         'first_neuron': [4, 8, 16, 32, 64],
         'hidden_layers': [0, 1, 2, 3, 4],
         'batch_size': (2, 30, 10),
         'epochs': [2],
         'dropout': (0, 0.5, 5),
         'weight_regulizer': [None],
         'emb_output_dims':  [None],
         'shapes': ['brick', 'triangle', 0.2],
         'optimizer': [Adam, Nadam],
         'losses': [logcosh, categorical_crossentropy],
         'activation': [relu, elu],
         'last_activation': [softmax]}

    return p


def breast_cancer():

    from tensorflow.keras.optimizers import Adam, Nadam, RMSprop
    from tensorflow.keras.losses import logcosh, binary_crossentropy
    from tensorflow.keras.activations import relu, elu, sigmoid

    # then we can go ahead and set the parameter space
    p = {'lr': (0.5, 5, 10),
         'first_neuron': [4, 8, 16, 32, 64],
         'hidden_layers': [0, 1, 2],
         'batch_size': (2, 30, 10),
         'epochs': [50, 100, 150],
         'dropout': (0, 0.5, 5),
         'shapes': ['brick', 'triangle', 'funnel'],
         'optimizer': [Adam, Nadam, RMSprop],
         'losses': [logcosh, binary_crossentropy],
         'activation': [relu, elu],
         'last_activation': [sigmoid]}

    return p


def cervical_cancer():
    return breast_cancer()
