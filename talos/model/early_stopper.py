from keras.callbacks import EarlyStopping


def early_stopper(epochs=None,
                  monitor='val_loss',
                  mode='moderate',
                  min_delta=None,
                  patience=None):

    '''EARLY STOP CALLBACK

    Helps prevent wasting time when loss is not becoming
    better. Offers two pre-determined settings 'moderate'
    and 'strict' and allows input of list with two values:

    `epochs` | int | The number of epochs for the permutation e.g. params['epochs']
    `monitor` | int | The metric to monitor for change
    `mode` | str | One of the presets `lazy`, `moderate`, `strict` or `None`
    `min_delta` | float | The limit for change at which point flag is raised
    `patience` | str | the number of epochs before termination from flag

    '''
    if mode == 'lazy':
        _es_out = EarlyStopping(monitor=monitor,
                                min_delta=0,
                                patience=int(epochs / 3),
                                verbose=0, mode='auto')

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
    else:
        _es_out = EarlyStopping(monitor=monitor,
                                min_delta=mode[0],
                                patience=mode[1],
                                verbose=0, mode='auto')
    return _es_out
