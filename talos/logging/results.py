def run_round_results(self, out):

    '''Called from logging/logging_run.py

    THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUNDself.
    Takes in the history object from model.fit() and handles it.

    NOTE: The epoch level data will be dropped here each round.

    '''

    self._round_epochs = len(list(out.history.values())[0])

    _round_result_out = [self._round_epochs]

    # record the last epoch result
    for key in out.history.keys():
        _round_result_out.append(out.history[key][-1])

    # record the round hyper-parameters
    for key in self.round_params.keys():
        _round_result_out.append(self.round_params[key])

    return ",".join(str(i) for i in _round_result_out)


def save_result(self):

    '''SAVES THE RESULTS/PARAMETERS TO A CSV SPECIFIC TO THE EXPERIMENT'''

    import numpy as np

    np.savetxt(self.experiment_name + '.csv',
               self.result,
               fmt='%s',
               delimiter=',')


def result_todf(self):

    '''ADDS A DATAFRAME VERSION OF THE RESULTS TO THE CLASS OBJECT'''

    import pandas as pd

    self.result = pd.DataFrame(self.result)
    self.result.columns = self.result.iloc[0]
    self.result = self.result.drop(0)

    return self


def peak_epochs_todf(self):

    import pandas as pd

    return pd.DataFrame(self.peak_epochs, columns=self.peak_epochs[0]).drop(0)
