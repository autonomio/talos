from keras.layers import Dense, Dropout


def hidden_layers(model, params):

    for i in range(params['hidden_layers']):
        model.add(Dense(params['first_neuron'], activation=params['activation']))
        model.add(Dropout(0))
