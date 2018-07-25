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

    no_of_items = len(list(history.history.keys()))

    if no_of_items == 1:
        loss_entropy = entropy(history.history['loss'])
        acc_entropy = nan

    else:
        if 'acc' in history.history.keys():
            acc_entropy = entropy(history.history['val_acc'],
                                  history.history['acc'])
            loss_entropy = entropy(history.history['val_loss'],
                                   history.history['loss'])

        elif 'loss' in history.history.keys():
            loss_entropy = entropy(history.history['val_loss'],
                                   history.history['loss'])
            acc_entropy = nan

    return [acc_entropy, loss_entropy]
