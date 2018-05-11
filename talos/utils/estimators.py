import datetime as dt


def time_estimator(self):

    '''ESTIMATE DURATION'''

    if self.round_counter == 0:

        start = dt.datetime.now()
        self._model()
        end = dt.datetime.now()

        total = (end - start).seconds
        total = total * len(self.param_grid)

        print("%d scans will take roughly %d seconds" % (len(self.param_grid), total))
