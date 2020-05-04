def scan_run(self):

    '''The high-level management of the scan procedures
    onwards from preparation. Manages round_run()'''

    import yaml
    import pickle
    from os.path import isfile
    from tqdm import tqdm

    from .scan_prepare import scan_prepare
    self = scan_prepare(self)

    # initiate the progress bar
    if self.allow_resume:
        self.pbar = tqdm(total=max(self.param_object.param_index)+1,
                         disable=self.disable_progress_bar,
                         initial=self.param_object.param_index[0])
    else:
        self.pbar = tqdm(total=len(self.param_object.param_index),
                         disable=self.disable_progress_bar)

    # the main cycle of the experiment
    while True:

        # get the parameters
        self.round_params = self.param_object.round_parameters()

        # break when there is no more permutations left
        if self.round_params is False:
            break
        # otherwise proceed with next permutation
        from .scan_round import scan_round
        self = scan_round(self)
        # Dump keys to backup file
        _keys_file = self._config_file.replace('yaml', 'keys.yaml')
        if self.allow_resume and not isfile(_keys_file):
            with open(_keys_file, 'w') as keys_object:
                yaml.dump((self._all_keys, self._metric_keys, self._val_keys), keys_object)
        # dump ParamSpace and various stores to backup files
        if self.allow_resume:
            with open(self._config_file, 'w') as file_object:
                yaml.dump(self.param_object, file_object, default_flow_style=False)
            with open(self._pickle_file, "wb") as p_file:
                pickle.dump(self.stores, p_file)
        self.pbar.update(1)

    # close progress bar before finishing
    self.pbar.close()

    # finish
    from ..logging.logging_finish import logging_finish
    self = logging_finish(self)

    from .scan_finish import scan_finish
    self = scan_finish(self)
