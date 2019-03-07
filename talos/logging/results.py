def create_header(self):

    '''Called from logging/logging_run.py

    Creates a header to the results table on
    the first round of the experiment before
    logging any results.
    '''

    _results_header = []

    _results_header.append('round_epochs')
    [_results_header.append(i) for i in self._all_keys]
    [_results_header.append(key) for key in self.params.keys()]

    return ",".join(str(i) for i in _results_header)


def run_round_results(self, out):

    '''Called from logging/logging_run.py

    THE MAIN FUNCTION FOR CREATING RESULTS FOR EACH ROUNDself.
    Takes in the history object from model.fit() and handles it.

    NOTE: The epoch level data will be dropped here each round.

    '''

    import numpy as np

    _rr_out = []

    self._round_epochs = len(list(out.history.values())[0])

    # otherwise proceed to create the value row
    _rr_out.append(self._round_epochs)
    p_epochs = []

    # iterates through the keys and records last or peak for metrics
    for key in out.history.keys():
        t_t = np.array(out.history[key])

        # this handles metrics (NOTE: 'acc' have to be in metric name)
        if 'acc' in key:
            best_epoch = np.argpartition(t_t, len(t_t) - 1)[-1]

        # this handles losses (takes minimum value epoch)
        else:
            best_epoch = np.argpartition(t_t, 0)[0]

        if self.last_epoch_value:
            value_to_report = out.history[key][-1]
        else:
            value_to_report = np.array(out.history[key])[best_epoch]

        _rr_out.append(value_to_report)
        p_epochs.append(best_epoch)

        # this takes care of the separate entity with just peak epoch data
    self.peak_epochs.append(p_epochs)

    for key in self.round_params.keys():
        _rr_out.append(self.round_params[key])

    return ",".join(str(i) for i in _rr_out)


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
