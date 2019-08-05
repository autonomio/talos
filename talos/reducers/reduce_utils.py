def cols_to_multilabel(self):

    '''Utility function for correlation and other reducers
    that require transforming hyperparameter values into
    multilabel values before applying the reduction strategy.'''

    import wrangle
    import pandas as pd

    # read in the experiment log
    data = pd.read_csv(self._experiment_log)

    # apply recuction window
    data = data.tail(self.reduction_window)

    # drop all other metric columns except reduction_metric
    data = data[[self.reduction_metric] + self._param_dict_keys]

    # convert all hyperparameter columns to multi label columns
    for col in data.iloc[:, 1:].columns:

        # get the dtype of the column data
        col_dtype = data[col].dtype

        # parse column name to contain label, value and dtype
        data = wrangle.col_to_multilabel(data,
                                         col,
                                         extended_colname=True,
                                         extended_separator='~' + str(col_dtype) + '~')

    return data
