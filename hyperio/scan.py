from keras import backend as K

from .utils.validation_split import validation_split

from .utils.results import run_round_results, save_result, result_todf
from .utils.logging import write_log
from .utils.detector import prediction_type
from .reducers.sample_reducer import sample_reducer
from .reducers.spear_reducer import spear_reducer
from .utils.estimators import time_estimator
from .parameters.handling import param_format, param_space, param_index, round_params
from .parameters.permutations import param_grid
from .utils.save_load import save_model
from .metrics.score_model import get_score


class Hyperio:

    global self

    def __init__(self, x, y, params, dataset_name, experiment_no, model,
                 val_split=.3, shuffle=True, search_method='random',
                 save_best_model=False,
                 reduction_method=None, reduction_interval=100,
                 reduction_window=None, grid_downsample=None,
                 hyperio_log_name='hyperio.log', debug=False):

        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = dataset_name + '_' + experiment_no

        if debug == True:
            self.logfile = open('hyperio.debug.log', 'a')
        else:
            self.logfile_name = hyperio_log_name
            self.logfile = open(self.logfile_name, 'a')

        self.model = model
        self.param_dict = params

        self.search_method = search_method
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.grid_downsample = grid_downsample
        self.val_split = val_split
        self.shuffle = shuffle
        self.save_model = save_best_model

        self.p = param_format(self)
        self.combinations = param_space(self)
        self.param_grid = param_grid(self)
        self.param_grid = sample_reducer(self)
        self.param_log = list(range(len(self.param_grid)))
        self.param_grid = param_index(self)
        self.round_counter = 0

        self.x = x
        self.y = y
        self = validation_split(self)

        self._data_len = len(self.x)
        self = prediction_type(self)

        self.result = []
        while len(self.param_log) != 0:
            self._null = self._run()

        self = result_todf(self)
        self._null = self.logfile.close()
        print('Scan Finished!')

    def _run(self):

        round_params(self)
        _hr_out, self.keras_model = self._model()
        _hr_out = run_round_results(self, _hr_out)

        self._val_score = get_score(self)

        write_log(self)
        self.result.append(_hr_out)
        save_result(self)
        time_estimator(self)

        if (self.round_counter + 1) % self.reduction_interval == 0:
            if self.reduction_method == 'spear':
                self = spear_reducer(self)

        if self.save_model == True:
            save_model(self)

        K.clear_session()
        self.round_counter += 1


    def _model(self):

        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)
