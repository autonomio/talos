from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop
from keras.optimizers import Nadam


def lr_normalizer(lr, optimizer):
    '''NORMALIZE LEARNING RATE ON DEFAULT 1'''

    if optimizer == Adadelta:
        pass
    elif optimizer == SGD or optimizer == Adagrad:
        lr /= 100.0
    elif optimizer == Adam or optimizer == RMSprop:
        lr /= 1000.0
    elif optimizer == Adamax or optimizer == Nadam:
        lr /= 500.0

    return lr
