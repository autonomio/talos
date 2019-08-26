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

    elif isinstance(task, list):
        try:
            # multilabel
            activation = last_activation
            last_neuron = y_train.shape[1]
        except IndexError:
            uniques = np.unique(np.hstack((y_train, y_val)))
            # binary
            if uniques == 2:
                activation = last_activation
                last_neuron = 1
            # multiclass (note this supports only < 10 classes)
            elif uniques <= 10:
                activation = last_activation
                last_neuron = np.unique(np.hstack((y_train, y_val)))
            # continuous maybe, or too many classes
            else:
                activation = None
                last_neuron = 1

    return activation, last_neuron
