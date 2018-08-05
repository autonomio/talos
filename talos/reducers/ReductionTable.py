import pandas as pd

from ..metrics.names import metric_names


class ReductionTable:

    '''Create the input for Reducers()'''

    def __init__(self,
                 filename,
                 reduction_metric,
                 reduction_window,
                 reduction_threshold):

        '''Takes as input the experiment results .csv and
        returns an object that can be used in Reduction()'''

        # load the data from the experiment log
        self.data = pd.read_csv(filename)

        # set the paramaters
        self.reduction_window = reduction_window
        self.reduction_metric = reduction_metric
        self.reduction_threshold = reduction_threshold
        self.names = metric_names()

        # apply the lookback window
        if self.reduction_window is not None:
            self.data = self.data.tail(self.reduction_window)

        # execute
        self._null = self.runtime()

    def runtime(self):

        self.param_columns = self._param_cols()
        self.param_table = self._create_table()
        self._null = self._merge_with_measure()

    def _param_cols(self):

        return [col for col in self.data.columns if col not in self.names]

    def _create_table(self):

        return self.data[self.param_columns]

    def _merge_with_measure(self):

        self.param_table.insert(0,
                                self.reduction_metric,
                                self.data[self.reduction_metric])
