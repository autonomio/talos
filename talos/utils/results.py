from numpy import array, argpartition, savetxt
from pandas import DataFrame


def create_header(self, out):

    '''Creates the Header column
    On the first round creates the header columns
    for the experiment output log.
    '''

    _rr_out = []

    _rr_out.append('round_epochs')
    [_rr_out.append(i) for i in list(out.history.keys())]
    [_rr_out.append(key) for key in self.params.keys()]

    self.peak_epochs.append(list(out.history.keys()))

    return ",".join(str(i) for i in _rr_out)


def run_round_results(self, out):

    '''THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUND

    NOTE: The epoch level data will be dropped here each round.

    '''

    _rr_out = []

    self._round_epochs = len(list(out.history.values())[0])

    # otherwise proceed to create the value row
    _rr_out.append(self._round_epochs)
    p_epochs = []

    for key in out.history.keys():
        t_t = array(out.history[key])

        # note that this requires all metrics / custom metrics
        # to include 'acc'
        if 'acc' in key:
            peak_epoch = argpartition(t_t, len(t_t) - 1)[-1]

        else:
            peak_epoch = argpartition(t_t, len(t_t) - 1)[0]

        peak = array(out.history[key])[peak_epoch]
        _rr_out.append(peak)
        p_epochs.append(peak_epoch)

        # this takes care of the separate entity with just peak epoch data
    self.peak_epochs.append(p_epochs)

    for key in self.round_params.keys():
        _rr_out.append(self.round_params[key])

    return ",".join(str(i) for i in _rr_out)


def save_result(self):
    '''SAVES THE RESULTS/PARAMETERS TO A CSV SPECIFIC TO THE EXPERIMENT'''

    savetxt(self.experiment_name + '.csv',
            self.result,
            fmt='%s',
            delimiter=',')


def result_todf(self):
    '''ADDS A DATAFRAME VERSION OF THE RESULTS TO THE CLASS OBJECT'''

    self.result = DataFrame(self.result)
    self.result.columns = self.result.iloc[0]
    self.result = self.result.drop(0)

    return self


def peak_epochs_todf(self):

    return DataFrame(self.peak_epochs, columns=self.peak_epochs[0]).drop(0)
