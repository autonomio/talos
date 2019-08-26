def scan_prepare(self):

    '''Includes all preparation procedures up until starting the first scan
    through scan_run()'''

    from .scan_utils import initialize_log

    self._experiment_log = initialize_log(self)

    # for the case where x_val or y_val is missing when other is present
    self.custom_val_split = False
    if (self.x_val is not None and self.y_val is None) or \
       (self.x_val is None and self.y_val is not None):
        raise RuntimeError("If x_val/y_val is inputted, other must as well.")

    elif self.x_val is not None and self.y_val is not None:
        self.custom_val_split = True

    # create reference for parameter keys
    self._param_dict_keys = sorted(list(self.params.keys()))

    # create the parameter object and move to self
    from ..parameters.ParamSpace import ParamSpace
    self.param_object = ParamSpace(params=self.params,
                                   param_keys=self._param_dict_keys,
                                   random_method=self.random_method,
                                   fraction_limit=self.fraction_limit,
                                   round_limit=self.round_limit,
                                   time_limit=self.time_limit,
                                   boolean_limit=self.boolean_limit
                                   )

    # mark that it's a first round
    self.first_round = True

    # create various stores
    self.round_history = []
    self.peak_epochs = []
    self.epoch_entropy = []
    self.round_times = []
    self.result = []
    self.saved_models = []
    self.saved_weights = []

    # handle validation split
    from ..utils.validation_split import validation_split
    self = validation_split(self)

    # set data and len
    self._data_len = len(self.x)

    return self
