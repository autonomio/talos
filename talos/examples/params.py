from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import relu, elu, softmax, sigmoid
from keras.losses import logcosh, binary_crossentropy, categorical_crossentropy


def iris():

    # here use a standard 2d dictionary for inputting the param boundaries
    p = {'lr': (0.5, 5, 10),
         'first_neuron': [4, 8, 16, 32, 64],
         'hidden_layers': [0, 1, 2, 3, 4],
         'batch_size': (2, 30, 10),
         'epochs': [2],
         'dropout': (0, 0.5, 5),
         'weight_regulizer': [None],
         'emb_output_dims':  [None],
         'shape': ['brick', 'rhombus'],
         'optimizer': [Adam, Nadam],
         'losses': [logcosh, categorical_crossentropy],
         'activation': [relu, elu],
         'last_activation': [softmax]}

    return p


def breast_cancer():

    # then we can go ahead and set the parameter space
    p = {'lr': (0.5, 5, 10),
         'first_neuron': [4, 8, 16, 32, 64],
         'hidden_layers': [0, 1, 2],
         'batch_size': (2, 30, 10),
         'epochs': [50, 100, 150],
         'dropout': (0, 0.5, 5),
         'weight_regulizer': [None],
         'emb_output_dims': [None],
         'shape': ['brick', 'long_funnel'],
         'optimizer': [Adam, Nadam, RMSprop],
         'losses': [logcosh, binary_crossentropy],
         'activation': [relu, elu],
         'last_activation': [sigmoid]}

    return p
