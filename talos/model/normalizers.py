from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam


def lr_normalizer(lr, optimizer):

    '''NORMALIZE LEARNING RATE ON DEFAULT 1'''

    if optimizer == Adadelta:
        lr = lr
    elif optimizer == SGD:
        lr = lr / 100
    elif optimizer == Adam:
        lr = lr / 1000
    elif optimizer == Adagrad:
        lr = lr / 100
    elif optimizer == Adamax:
        lr = lr / 500
    elif optimizer == RMSprop:
        lr = lr / 1000
    elif optimizer == Nadam:
        lr = lr / 500

    return lr
