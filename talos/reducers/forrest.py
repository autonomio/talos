def forrest(self):

    '''Random Forrest based reduction strategy. Somewhat more
    aggressive than for example 'spearman' because there are no
    negative values, but instead the highest positive correlation
    is minused from all the values so that max value is 0, and then
    values are turned into positive. The one with the highest positive
    score in the end will be dropped. This means that anything with
    0 originally, is a candidate for dropping. Because there are multiple
    zeroes in many cases, there is an element of randomness on which one
    is dropped.

    '''

    import wrangle
    import numpy as np

    # handle conversion to multi_labels
    from .reduce_utils import cols_to_multilabel
    data = cols_to_multilabel(self)

    # get the correlations
    corr_values = wrangle.df_corr_randomforest(data, self.reduction_metric)

    # drop labels where value is NaN
    corr_values.dropna(inplace=True)

    # handle the turning around of values (see docstring for more info)
    corr_values -= corr_values[0]
    corr_values = corr_values.abs()

    # get the strongest correlation
    corr_values = corr_values.index[-1]

    # get the label, value, and dtype from the column header
    label, dtype, value = corr_values.split('~')

    # convert things back to their original dtype
    value = np.array([value]).astype(dtype)[0]

    # this is where we modify the parameter space accordingly
    self.param_object.remove_is(label, value)

    return self
