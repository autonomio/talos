def logging_run(self, round_start, start, model_history):

    import time

    # count the duration of the round
    self._round_seconds = time.time() - start

    # set end time and log
    round_end = time.strftime('%D-%H%M%S')
    self.round_times.append([round_start, round_end, self._round_seconds])

    # handle first round only things
    if self.first_round:

        # capture the history keys for later
        self._all_keys = list(model_history.history.keys())
        self._metric_keys = [k for k in self._all_keys if 'val_' not in k]
        self._val_keys = [k for k in self._all_keys if 'val_' in k]

        # create a header column for output
        _results_header = ['round_epochs'] + self._all_keys + self._param_dict_keys
        self.result.append(_results_header)

        # save the results
        from .results import save_result
        save_result(self)

        # avoid doing this again
        self.first_round = False

    # create log and other stats
    from ..metrics.entropy import epoch_entropy
    self.epoch_entropy.append(epoch_entropy(self, model_history.history))

    # get round results to the results table and save it
    from .results import run_round_results
    _round_results = run_round_results(self, model_history)

    self.result.append(_round_results)

    from .results import save_result
    save_result(self)

    # return the Scan() self
    return self
