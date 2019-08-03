def scan_finish(self):

    attrs_final = ['data', 'x', 'y', 'learning_entropy', 'round_times',
                   'params', 'saved_models', 'saved_weights', 'round_history']

    attrs_to_keep = attrs_final + ['random_method', 'grid_downsample',
                                   'reduction_interval', 'reduce_loss',
                                   'reduction_method', 'reduction_metric',
                                   'reduction_threshold', 'reduction_window',
                                   'experiment_name', 'round_history']

    import time
    import pandas as pd

    # create a dataframe with permutation times
    self.round_times = pd.DataFrame(self.round_times)
    self.round_times.columns = ['start', 'end', 'duration']

    # combine epoch entropy tables
    self.learning_entropy = pd.DataFrame(self.epoch_entropy)
    self.learning_entropy.columns = self._metric_keys

    # clean the results into a dataframe
    self.data = self.result

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
    try:
        out['x_shape'] = self.x.shape
    # for the case when x is list
    except AttributeError:
        out['x_shape'] = 'list'

    try:
        out['y_shape'] = self.y.shape
    except AttributeError:
        out['y_shape'] = 'list'

    # final cleanup
    keys = list(self.__dict__.keys())
    for key in keys:
        if key not in attrs_final:
            delattr(self, key)

    # add details dictionary as series
    self.details = pd.Series(out)

    # add best_model

    from ..scan.scan_addon import func_best_model, func_evaluate

    self.best_model = func_best_model.__get__(self)
    self.evaluate_models = func_evaluate.__get__(self)

    # reset the index
    self.data.index = range(len(self.data))

    return self
