import pandas as pd


def correlation(self):

    '''Correlation Reducers

    Note that this set of reducers works only for the continuous
    and stepped (e.g. batch size) hyperparameters.

    '''

    out = self.param_table.corr(method='spearman')[self.reduction_metric]
    out = out.dropna()

    if len(out) == 0:
        self._reduce_keys = None
        return self

    out = out[1:].sort_values(ascending=False)
    out = out.index[-1], out[-1]

    if abs(out[1]) >= self.reduction_threshold:
        dummy_cols = pd.get_dummies(self.param_table[out[0]])
        dummy_cols.insert(0,
                          self.reduction_metric,
                          self.param_table[self.reduction_metric])

    # case where threshold is not met
    else:
        self._reduce_keys = None
        return self

    # all other cases continue
    to_drop_temp = dummy_cols.corr(method='spearman')[self.reduction_metric]

    # pick the drop method based on paramaters
    if self.reduce_loss is False:
        self._reduce_keys = to_drop_temp.sort_values().index[0], out[0]
    else:
        self._reduce_keys = to_drop_temp.sort_values().index[-2], out[0]

    return self
