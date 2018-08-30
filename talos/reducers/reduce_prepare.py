import pandas as pd

from ..metrics.names import metric_names


def reduce_prepare(self):

    '''
    Preparation procedures for applying a reduction algorithm.
    '''

    # load the data from the experiment log
    self.data = pd.read_csv(self.experiment_name + '.csv')
    self.names = metric_names()

    # apply the lookback window
    if self.reduction_window is not None:
        self.data = self.data.tail(self.reduction_window)

    self.param_columns = [col for col in self.data.columns if col not in metric_names()]
    self.param_table = self.data[self.param_columns]
    self.param_table.insert(0, self.reduction_metric, self.data[self.reduction_metric])

    return self
