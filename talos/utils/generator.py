"""Yield caller-array batches through the historical training generator."""


def generator(x, y, batch_size):

    '''Creates a data generator for Keras fit_generator(). '''

    import numpy as np

    number_of_batches = x.shape[0] / batch_size
    counter = 0

    while 1:

        x_batch = np.array(x[batch_size * counter:batch_size * (counter + 1)]).astype('float32')
        y_batch = np.array(y[batch_size * counter:batch_size * (counter + 1)]).astype('float32')
        counter += 1

        yield x_batch, y_batch

        if counter >= number_of_batches:
            counter = 0
