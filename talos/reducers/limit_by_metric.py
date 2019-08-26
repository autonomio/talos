def limit_by_metric(self):

    '''Takes as input metric, threshold, and loss and
    and returs a True if metric threshold have been
    met and False if not.

    USE: space.check_metric(model_history)
    '''

    metric = self.performance_target[0]
    threshold = self.performance_target[1]
    loss = self.performance_target[2]

    if loss is True:
        return self.model_history.history[metric][-1] <= threshold
    elif loss is False:
        return self.model_history.history[metric][-1] >= threshold
