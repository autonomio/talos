def scan_prepare(self):

    '''Includes all preparation procedures up until starting the first scan
    through scan_run()'''

    import time as ti

    # create the name for the experiment
    if self.experiment_name is None:
        self.experiment_name = ti.strftime('%D%H%M%S').replace('/', '')

    # for the case where x_val or y_val is missing when other is present
    self.custom_val_split = False
    if (self.x_val is not None and self.y_val is None) or \
       (self.x_val is None and self.y_val is not None):
        raise RuntimeError("If x_val/y_val is inputted, other must as well.")

    elif (self.x_val is not None and self.y_val is not None):
        self.custom_val_split = True

    # create the paramater object and move to self
    from ..parameters.ParamSpace import ParamSpace
    self.param_object = ParamSpace(params=self.params,
                                   random_method=self.random_method,
                                   fraction_limit=self.fraction_limit,
                                   round_limit=self.round_limit,
                                   time_limit=self.time_limit,
                                   boolean_limit=self.boolean_limit
                                   )

    # create various stores
    self.round_history = []
    self.peak_epochs = []
    self.epoch_entropy = []
    self.round_times = []
    self.result = []
    self.saved_models = []
    self.saved_weights = []

    # create the data asset
    self.y_max = self.y.max()

    # handle validation split
    from ..utils.validation_split import validation_split
    self = validation_split(self)

    # set data and len
    self.shape = [self.x.shape, self.y.shape]
    self._data_len = len(self.x)

    # infer prediction type
    from ..utils.detector import prediction_type
    self = prediction_type(self)

    return self
