from keras import backend as K

from .utils.template import df_cols
from .utils.validation_split import validation_split

from .utils.results import run_round_results, save_result, result_todf
from .utils.logging import write_log
from .utils.detector import prediction_type
from .reducers.sample_reducer import sample_reducer
from .reducers.spear_reducer import spear_reducer
from .utils.estimators import time_estimator
from .parameters.handling import param_format, param_space, param_index, round_params
from .parameters.permutations import param_grid


class Hyperio:

    global self

    def __init__(self, x, y, params, dataset_name, experiment_no, model,
                 val_split=.3, shuffle=True, search_method='random',
                 reduction_method=None, reduction_interval=100,
                 reduction_window=None,
                 grid_downsample=None, hyperio_log_name='hyperio.log',
                 debug=False):

        # experiment name
        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = dataset_name + '_' + experiment_no

        # logfile initialization
        if debug == True:
            self.logfile = open('hyperio.debug.log', 'a')
        else:
            self.logfile_name = hyperio_log_name
            self.logfile = open(self.logfile_name, 'a')

        # load params dictionary and model
        self.model = model
        self.param_dict = params

        # load input parameters
        self.search_method = search_method
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.grid_downsample = grid_downsample
        self.val_split = val_split
        self.shuffle = shuffle
        self.df_col_list = df_cols()


        # prepare the parameter search boundary
        self.p = param_format(self)
        self.combinations = param_space(self)
        self.param_grid = param_grid(self)
        self.param_grid = sample_reducer(self)
        self.param_log = list(range(len(self.param_grid)))
        self.param_grid = param_index(self)
        self.round_counter = 0

        # prepare data
        self.x = x
        self.y = y
        self = validation_split(self)

        # create data related log data
        self._data_len = len(self.x)
        self = prediction_type(self)

        # run the scan
        self.result = []

        while len(self.param_log) != 0:
            self._null = self._run()

        # get the results ready
        self = result_todf(self)

        # close the log file
        self._null = self.logfile.close()
        print('Scan Finished!')

    # THE MAIN RUNTIME FUNCTION STARTS
    # --------------------------------
    def _run(self):

        '''RUNTIME'''

        round_params(self)    # this creates the params round

        print(self.params)

        _hr_out = self._model()
        _hr_out = run_round_results(self, _hr_out)
        write_log(self)
        self.result.append(_hr_out)
        save_result(self)

        # this is for the first round only
        time_estimator(self)

        # reducing algorithsm
        if (self.round_counter + 1) % self.reduction_interval == 0:

            if self.reduction_method == 'spear':
                self = spear_reducer(self)

        # prevent Tensorflow memory leakage
        K.clear_session()

        # add one to the total round counter
        self.round_counter += 1

    # ------------------------------
    # THE MAIN RUNTIME FUNCTION ENDS

    def _model(self):

        '''RUNS THE USERS MODEL

        This is loaded from the actual user Hyperio call from
        model= parameter.

        '''

        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)
