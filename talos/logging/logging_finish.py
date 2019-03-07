def logging_finish(self):

    from .results import result_todf, peak_epochs_todf

    # save the results
    self = result_todf(self)
    self.peak_epochs = peak_epochs_todf(self)

    return self
