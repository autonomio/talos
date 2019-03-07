def correlation(self):

    '''This is called from reduce_run.py.

    Performs a spearman rank order correlation
    based reduction. First looks for a parameter
    that correlates with reduction_metric and
    correlation meets reduction_threshold and
    then converts the match parameter into
    a 2d multilabel shape. Then new correlation
    against reduction_metric is performed to identify
    which particular value is to be dropped.

    '''
    import pandas as pd
    import wrangle as wr

    data = pd.read_csv(self.experiment_name + '.csv')
    data = data[[self.reduction_metric] + self.param_object.column_names]

    corr = data.copy(deep=True)

    # drop the row for reduction metric and sort
    corr = corr.dropna()
    corr = data.corr('spearman')
    corr = corr[self.reduction_metric]
    corr = corr.apply(abs)[1:]
    corr = corr.sort_values(ascending=False)

    # check if reduction threshold is met:
    if corr.values[0] <= self.reduction_threshold is self.minimize_loss:
        return False

    # filter out where only one value is present
    if len(corr) <= 1:
        self._reduce_keys = None
        return False

    label = corr.index.values[0]
    if label not in self.param_object.column_names:
        return False

    # convert parameter values to multilabel (2d)
    corr = wr.col_to_multilabel(data[[label]], label)

    # combine the reduction_metric with the multilabel data
    corr = wr.df_merge(corr, data[self.reduction_metric])

    # repeate same as above
    corr = corr.corr('spearman')
    corr = corr[self.reduction_metric]
    corr = corr.apply(abs)[1:]
    corr = corr.sort_values(ascending=False)

    if len(corr) <= 1:
        self._reduce_keys = None
        return False

    label = corr.index.values[0]
    if label not in self.param_object.column_names:
        return False

    value = corr.values[0]
    label = corr.index[0]

    return value, label
