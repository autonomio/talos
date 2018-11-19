from keras.callbacks import EarlyStopping


def early_stopper(epochs,
                  monitor='val_loss',
                  mode='moderate',
                  min_delta=None,
                  patience=None):

    '''EARLY STOP CALLBACK

    Helps prevent wasting time when loss is not becoming
    better. Offers two pre-determined settings 'moderate'
    and 'strict' and allows input of list with two values:

    min_delta = the limit for change at which point flag is raised

    patience = the number of epochs before termination from flag

    '''

    if mode == 'moderate':
        _es_out = EarlyStopping(monitor=monitor,
                                min_delta=0,
                                patience=int(epochs / 10),
                                verbose=0, mode='auto')
    elif mode == 'strict':
        _es_out = EarlyStopping(monitor=monitor,
                                min_delta=0,
                                patience=2,
                                verbose=0, mode='auto')
    elif isinstance(mode, list):
        _es_out = EarlyStopping(monitor=monitor,
                                min_delta=mode[0],
                                patience=mode[1],
                                verbose=0, mode='auto')
    return _es_out
