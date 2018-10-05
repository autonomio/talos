from scipy.stats import entropy
from numpy import nan


def epoch_entropy(history):

    '''MEASURE EPOCH ENTROPY

    BINARY/CATEGORICAL:

    Measures the KL divergence of the acc and loss results
    per epoch of a given permutation.

    CONTINUOUS:

    Measures shannon entropy for loss.

    # TODO Right now this does not handle all cases well and needs
      to be thought about properly.
    '''

    keys = list(history.history.keys())
    no_of_items = len(keys)

    if no_of_items == 1:
        if 'loss' in keys:
            loss_entropy = entropy(history.history['loss'])
            acc_entropy = nan
        else:
            loss_entropy = nan
            acc_entropy = nan

    elif no_of_items == 2:
        if 'acc' in keys and 'loss' in keys:
            loss_entropy = entropy(history.history['loss'])
            acc_entropy = entropy(history.history['acc'])
        else:
            loss_entropy = nan
            acc_entropy = nan

    elif no_of_items >= 4:
        if 'acc' in keys:
            acc_entropy = entropy(history.history['val_acc'],
                                  history.history['acc'])
        else:
            acc_entropy = nan

        if 'loss' in keys:
            loss_entropy = entropy(history.history['val_loss'],
                                   history.history['loss'])
        else:
            loss_entropy = nan

    return [acc_entropy, loss_entropy]
