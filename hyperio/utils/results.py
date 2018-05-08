from numpy import array, argpartition, savetxt
from pandas import DataFrame

from .template import value_cols


def run_round_results(self, out):

    '''THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUND

    NOTE: The epoch level data will be dropped here each round.

    '''
    round_epochs = len(out.history['acc'])

    t_t = array(out.history['acc']) - array(out.history['loss'])
    v_t = array(out.history['val_acc']) - array(out.history['val_loss'])

    train_peak = argpartition(t_t, round_epochs-1)[-1]
    val_peak = argpartition(v_t, round_epochs-1)[-1]

    train_acc = array(out.history['acc'])[train_peak]
    train_loss = array(out.history['loss'])[train_peak]
    train_score = train_acc - train_loss

    val_acc = array(out.history['val_acc'])[val_peak]
    val_loss = array(out.history['val_loss'])[val_peak]
    val_score = val_acc - val_loss

    # this is for the log
    self._val_score = val_score
    self._round_epochs = round_epochs

    # if the round counter is zero, just output header
    if self.round_counter == 0:
        _rr_out = value_cols()
        [_rr_out.append(key) for key in self.params.keys()]
        return _rr_out

    # otherwise proceed to create the value row
    _rr_out = []
    for val in value_cols():
        _rr_out.append(locals()[val])

    # this is based on columns defined in template.py
    for key in self.params.keys():
        _rr_out.append(self.params[key])

    return _rr_out


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
