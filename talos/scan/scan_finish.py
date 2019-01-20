import time
from pandas import Series, DataFrame

from ..utils.best_model import best_model
from ..scan.scan_addon import func_best_model, func_evaluate

attrs_to_keep = ['data', 'x', 'y', 'peak_epochs_df',
                 'random_method', 'grid_downsample',
                 'reduction_interval', 'reduce_loss',
                 'reduction_method', 'reduction_metric',
                 'reduction_threshold', 'reduction_window',
                 'params', 'saved_models', 'saved_weights',
                 'experiment_name']

attrs_final = ['data', 'x', 'y', 'peak_epochs_df', 'round_times',
               'params', 'saved_models', 'saved_weights']


def scan_finish(self):

    # create a dataframe with permutation times
    self.round_times = DataFrame(self.round_times)
    self.columns = ['start', 'end', 'duration']

    # combine entropy tables
    self.peak_epochs_df['acc_epoch'] = [i[0] for i in self.epoch_entropy]
    self.peak_epochs_df['loss_epoch'] = [i[1] for i in self.epoch_entropy]

    # clean the results into a dataframe
    self.data = self.result[self.result.columns[0]].str.split(',', expand=True)
    self.data.columns = self.result.columns[0].split(',')

    # remove redundant columns
    keys = list(self.__dict__.keys())
    for key in keys:
        if key not in attrs_to_keep:
            delattr(self, key)

    # summarize single inputs in dictionary
    out = {}

    for key in list(self.__dict__.keys()):
        if key not in attrs_final:
            out[key] = self.__dict__[key]

    out['complete_time'] = time.strftime('%D/%H:%M')
    out['x_shape'] = self.x.shape
    out['y_shape'] = self.y.shape

    # final cleanup
    keys = list(self.__dict__.keys())
    for key in keys:
        if key not in attrs_final:
            delattr(self, key)

    # add details dictionary as series
    self.details = Series(out)

    # add best_model
    self.best_model = func_best_model.__get__(self)
    self.evaluate_models = func_evaluate.__get__(self)

    # reset the index
    self.data.index = range(len(self.data))

    return self
