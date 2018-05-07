from keras.optimizers import SGD, Adam, Adadelta, Adagrad, Adamax, RMSprop, Nadam
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout


def hidden_layers(model, params):

    for i in range(params['hidden_layers']):
        model.add(Dense(params['first_neuron'], activation=params['activation']))
        model.add(Dropout(0))


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


def early_stopper(epochs, mode='moderate', min_delta=None, patience=None):

    '''EARLY STOP CALLBACK

    Helps prevent wasting time when loss is not becoming
    better. Offers two pre-determined settings 'moderate'
    and 'strict' and allows input of list with two values:

    min_delta = the limit for change at which point flag is raised

    patience = the number of epochs before termination from flag

    '''

    if mode == 'moderate':
        _es_out = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=int(epochs / 10),
                                verbose=0, mode='auto')
    elif mode == 'strict':
        _es_out = EarlyStopping(monitor='val_loss',
                                min_delta=0,
                                patience=2,
                                verbose=0, mode='auto')
    elif type(mode) == type([]):
        _es_out = EarlyStopping(monitor='val_loss',
                                min_delta=min_delta,
                                patience=patience,
                                verbose=0, mode='auto')
    return [_es_out]
