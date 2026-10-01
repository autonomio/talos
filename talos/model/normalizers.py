def lr_normalizer(lr, optimizer):
    from talos.utils.exceptions import TalosModelError
    name = optimizer.__name__ if isinstance(optimizer, type) else type(optimizer).__name__
    divisors = {'SGD': 100, 'Adagrad': 100, 'Adam': 1000, 'RMSprop': 1000, 'Adamax': 500}
    if name not in divisors:
        raise TalosModelError(str(optimizer) + ' is not supported by lr_normalizer')
    return lr / divisors[name]
