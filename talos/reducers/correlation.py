def correlation(self, method):

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

    import numpy as np

    # transform the data properly first
    from .reduce_utils import cols_to_multilabel
    data = cols_to_multilabel(self)

    # get the correlations
    corr_values = data.corr(method)[self.reduction_metric]

    # drop the reduction metric row
    corr_values.drop(self.reduction_metric, inplace=True)

    # drop labels where value is NaN
    corr_values.dropna(inplace=True)

    # if all nans, then stop
    if len(corr_values) <= 1:
        return self

    # sort based on the metric type
    corr_values.sort_values(ascending=self.minimize_loss, inplace=True)

    # if less than threshold, then stop
    if abs(corr_values[-1]) < self.reduction_threshold:
        return self

    # get the strongest correlation
    corr_values = corr_values.index[-1]

    # get the label, value, and dtype from the column header
    label, dtype, value = corr_values.split('~')

    # convert things back to their original dtype
    value = np.array([value]).astype(dtype)[0]

    # this is where we modify the parameter space accordingly
    self.param_object.remove_is(label, value)

    return self
