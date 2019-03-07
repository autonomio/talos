def epoch_entropy(self, history):

    '''Called from logging/logging_run.py

    Computes the entropy for epoch metric
    variation. If validation is on,
    then returns KL divergence instead of
    simple Shannon entropy. When Keras
    validation_freq is on, Shannon entropy
    is returned. Basically, all experiments
    should use validation, so Shannon is
    provided mearly as a fallback.

    '''

    import warnings
    from scipy.stats import entropy

    warnings.simplefilter('ignore')

    out = []

    # set the default entropy mode to shannon
    mode = 'shannon'

    # try to make sure each metric has validation
    if len(self._metric_keys) == len(self._val_keys):
        # make sure that the length of the arrays are same
        for i in range(len(self._metric_keys)):
            if len(history[self._metric_keys[i]]) == len(history[self._val_keys[i]]):
                mode = 'kl_divergence'
            else:
                break

    # handle the case where only shannon entropy can be used
    if mode == 'shannon':
        for i in range(len(self._metric_keys)):
            out.append(entropy(history[self._metric_keys[i]]))

    # handle the case where kl divergence can be used
    elif mode == 'kl_divergence':
        for i in range(len(self._metric_keys)):
            out.append(entropy(history[self._val_keys[i]],
                               history[self._metric_keys[i]]))

    return out
