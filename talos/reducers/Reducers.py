import pandas as pd
from numpy import nan


class Reducers:

    '''A suite of methods for reducing time complexity of the scan.'''

    def __init__(self, reduction_table):

        '''Takes as input an object from ReductionTable()'''

        self.param_table = reduction_table.param_table
        self.objective_measure = reduction_table.reduction_metric
        self.reduction_threshold = reduction_table.reduction_threshold

    def correlation(self,
                    correlation='spearman',
                    corr_to_drop='neg'):

        '''Correlation Reducers

        Note that this set of reducers works only for the continuous
        and stepped (e.g. batch size) hyperparameters.

        '''

        out = self.param_table.corr(correlation)[self.objective_measure]
        out = out.dropna()

        if len(out) == 0:
            return None

        out = out[1:].sort_values(ascending=False)
        out = out.index[-1], out[-1]

        if abs(out[1]) >= self.reduction_threshold:
            dummy_cols = pd.get_dummies(self.param_table[out[0]])
            dummy_cols.insert(0,
                              self.objective_measure,
                              self.param_table[self.objective_measure])

        # case where threshold is not met
        else:
            return None

        # all other cases continue
        to_drop_temp = dummy_cols.corr(correlation)[self.objective_measure]

        # pick the drop method based on paramaters
        if corr_to_drop == 'neg':
            return to_drop_temp.sort_values().index[0], out[0]
        elif corr_to_drop == 'pos':
            return to_drop_temp.sort_values().index[-2], out[0]
