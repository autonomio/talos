def output_layer(task, last_activation, y_train, y_val):

    import numpy as np

    # output layer
    if task == 'binary':
        activation = last_activation
        last_neuron = 1

    elif task == 'multi_class':
        activation = last_activation
        last_neuron = len(np.unique(np.hstack((y_train, y_val))))

    elif task == 'multi_label':
        activation = last_activation
        last_neuron = y_train.shape[1]

    elif task == 'continuous':
        activation = None
        last_neuron = 1

    return activation, last_neuron
