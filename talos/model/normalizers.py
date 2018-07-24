from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.optimizers import Nadam


def lr_normalizer(lr, optimizer):
    """Assuming a default learning rate 1, rescales the learning rate
    such that learning rates amongst different optimizers are more or less
    equivalent.

    Parameters
    ----------
    lr : float
        The learning rate.
    optimizer : keras optimizer
        The optimizer. For example, Adagrad, Adam, RMSprop.
    """

    if optimizer == Adadelta:
        pass
    elif optimizer == SGD or optimizer == Adagrad:
        lr /= 100.0
    elif optimizer == Adam or optimizer == RMSprop:
        lr /= 1000.0
    elif optimizer == Adamax or optimizer == Nadam:
        lr /= 500.0

    return lr
