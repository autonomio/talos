def classify(y):

    '''Detects if prediction is binary, multi-label or multi-class'''

    shape = detect_shape(y)

    if shape > 1:
        return 'multi_class'

    elif y.max() <= 1:
        return 'binary_class'
    else:
        return 'multi_label'


def detect_shape(y):

    try:
        return y.shape[1]
    except IndexError:
        return 1
