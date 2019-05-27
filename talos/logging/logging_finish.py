def logging_finish(self):

    from .results import result_todf

    # save the results
    self = result_todf(self)

    return self
