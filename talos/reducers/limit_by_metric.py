def limit_by_metric(self):

    '''Takes as input metric, threshold, and loss and
    and returs a True if metric threshold have been
    met and False if not.

    USE: space.check_metric(model_history)
    '''

    temp = self.performance_target

    metric = temp[0]
    threshold = temp[1]
    loss = temp[2]

    if loss is True:
        return min(self.model_history[metric]) < threshold
    elif loss is False:
        return max(self.model_history[metric]) > threshold
