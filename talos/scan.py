from keras import backend as K
from tensorflow import get_default_graph, Session

from .utils.validation_split import validation_split

from .utils.results import run_round_results, save_result, result_todf, peak_epochs_todf
from .utils.logging import write_log
from .utils.detector import prediction_type
from .reducers.sample_reducer import sample_reducer
from .reducers.spear_reducer import spear_reducer
from .utils.estimators import time_estimator
from .parameters.handling import param_format, param_space, param_index, round_params
from .parameters.permutations import param_grid
from .metrics.score_model import get_score
from .utils.pred_class import classify
from .utils.last_neuron import last_neuron
from .metrics.entropy import epoch_entropy


class Scan:

    global self

    def __init__(self, x, y, params, dataset_name, experiment_no, model,
                 val_split=.3, shuffle=True, search_method='random',
                 reduction_method=None, reduction_interval=100,
                 reduction_window=None, grid_downsample=None,
                 reduction_metric='val_acc', round_limit=None,
                 talos_log_name='talos.log', debug=False):

        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = dataset_name + '_' + experiment_no

        if debug == True:
            self.logfile = open('talos.debug.log', 'a')
        else:
            self.logfile_name = talos_log_name
            self.logfile = open(self.logfile_name, 'a')

        self.model = model
        self.param_dict = params

        self.search_method = search_method
        self.reduction_method = reduction_method
        self.reduction_interval = reduction_interval
        self.reduction_window = reduction_window
        self.reduction_metric = reduction_metric
        self.grid_downsample = grid_downsample
        self.val_split = val_split
        self.shuffle = shuffle

        self.p = param_format(self)
        self.combinations = param_space(self)
        self.param_grid = param_grid(self)
        self.param_grid = sample_reducer(self)
        self.param_log = list(range(len(self.param_grid)))
        self.param_grid = param_index(self)
        self.round_counter = 0
        self.peak_epochs = []
        self.epoch_entropy = []
        self.round_limit = round_limit
        self.round_models = []

        self.x = x
        self.y = y
        self.y_max = y.max()
        self = validation_split(self)
        self.shape = classify(self.y)
        self.last_neuron = last_neuron(self)

        self._data_len = len(self.x)
        self = prediction_type(self)

        self.result = []

        if self.round_limit != None:
            for i in range(self.round_limit):
                self._null = self._run()
        else:
            while len(self.param_log) != 0:
                self._null = self._run()

        self = result_todf(self)
        self.peak_epochs_df = peak_epochs_todf(self)
        self._null = self.logfile.close()
        print('Scan Finished!')

    def _run(self):

        round_params(self)

        try:
            _hr_out, self.keras_model = self._model()
        except TypeError:
            print('The model needs to have Return in format "return history, model"')

        self.epoch_entropy.append(epoch_entropy((_hr_out)))
        _hr_out = run_round_results(self, _hr_out)

        self._val_score = get_score(self)

        write_log(self)
        self.result.append(_hr_out)
        save_result(self)
        time_estimator(self)

        if (self.round_counter + 1) % self.reduction_interval == 0:
            if self.reduction_method == 'spear':
                self = spear_reducer(self)

        K.clear_session()
        self.round_counter += 1

    def _model(self):

        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)
