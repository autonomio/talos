import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


def max_memory(fraction):

    '''Sets the max used memory as a fraction for tensorflow
    backend

    fraction :: a float value (e.g. 0.5 means 4gb out of 8gb)

    '''

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = fraction
    set_session(tf.Session(config=config))
