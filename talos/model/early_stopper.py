def early_stopper(epochs=None, monitor='val_loss', mode='moderate', min_delta=None,
                  patience=None, backend='keras'):
    if mode in ('lazy', 'moderate', 'strict'):
        if mode != 'strict' and epochs is None:
            raise ValueError('epochs is required for this early-stopping preset.')
        defaults = {'lazy': int((epochs or 0) / 3), 'moderate': int((epochs or 0) / 10), 'strict': 2}
        patience = defaults[mode] if patience is None else patience
        min_delta = 0 if min_delta is None else min_delta
    elif isinstance(mode, (list, tuple)) and len(mode) == 2:
        min_delta, patience = mode
    elif mode is None:
        min_delta = 0 if min_delta is None else min_delta
        patience = 0 if patience is None else patience
    else:
        raise ValueError('mode must be lazy, moderate, strict, None, or [min_delta, patience].')
    if backend in ('tensorflow', 'tf', 'tf.keras'):
        from tensorflow.keras.callbacks import EarlyStopping
    else:
        from keras.callbacks import EarlyStopping
    return EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=0, mode='auto')
