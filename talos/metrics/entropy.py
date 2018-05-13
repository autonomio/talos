from scipy.stats import entropy


def epoch_entropy(history):

    acc_entropy = entropy(history.history['val_acc'], history.history['acc'])
    loss_entropy = entropy(history.history['val_loss'], history.history['loss'])

    return [acc_entropy, loss_entropy]
