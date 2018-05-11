from numpy import array, argpartition, savetxt
from pandas import DataFrame


def run_round_results(self, out):

    '''THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUND

    NOTE: The epoch level data will be dropped here each round.

    '''

    _rr_out = []

    self._round_epochs = len(out.history['acc'])

    # if the round counter is zero, just output header
    if self.round_counter == 0:
        _rr_out.append('round_epochs')
        _rr_out = list(out.history.keys())
        [_rr_out.append(key) for key in self.params.keys()]
        return ",".join(str(i) for i in _rr_out)

    # otherwise proceed to create the value row
    _rr_out.append(self._round_epochs)
    for key in out.history.keys():
        t_t = array(out.history[key]) - array(out.history[key])
        peak_epoch = argpartition(t_t, self._round_epochs-1)[-1]
        peak_val = array(out.history[key])[peak_epoch]
        _rr_out.append(peak_val)

    for key in self.params.keys():
        _rr_out.append(self.params[key])

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
