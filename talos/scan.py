from keras import backend as K

from .utils.validation_split import validation_split

from .utils.results import run_round_results, save_result, result_todf
from .utils.results import peak_epochs_todf, create_header
from .utils.logging import write_log
from .utils.detector import prediction_type
from .reducers.spear_reducer import spear_reducer
from .parameters.round_params import round_params
from .parameters.ParamGrid import ParamGrid
from .metrics.score_model import get_score
from .utils.pred_class import classify
from .utils.last_neuron import last_neuron
from .metrics.entropy import epoch_entropy


TRAIN_VAL_RUNTIME_ERROR_MSG = """
If setting a custom train/val split, both x_val and y_val must be input data
and not None.
"""


class Scan:
    """Suite of operations for training and evaluating Keras neural networks.

    Inputs train/dev data and a set of parameters as a dictionary. The name and
    experiment number must also be chosen since they define the output
    filenames. The model must also be specified of the form

        my_model(x_train, y_train, x_val, y_val, params),

    and the dictionary

        d = {
            'fcc_layer_1_N': [50, 100, 200],
            'fcc_layer_1_act': ['relu', 'tanh'],
            'fcc_layer_1_dropout': (0, 0.1, 5)    # 5 points between 0 and 0.1
        }

    The dictionary is parsed for every run and only one entry per parameter
    is fed into the neural network at a time.

    Important note: the user has two options when specifying input data.

    Option 1:
        Specify x, y and val_split. The training and validation data mixture
        (x, y) will be randomly split into the training and validation datasets
        as per the split specified in val_split.

    Option 2:
        Specify x, y and x_val, y_val. This would allow the user to specify
        their own validation datasets. Keras by default shuffles data during
        training, so the user need only be sure that the split specified is
        correct. This allows for not only reproducibility, but randomizing the
        data on the user's own terms. This is critical if the user wishes to
        augment their training data without augmenting their validation data
        (which is the only acceptable practice!).


    Parameters
    ----------
    x : ndarray
        1d or 2d array consisting of the training data. `x` should have the
        shape (m, n), where m is the number of training examples and n is the
        number of features. Extra dimensions can be added to account for the
        channels entry in convolutional neural networks.
    y : ndarray
        The labels corresponding to the training data. `y` should have the
        shape (m, c) where c is the number of classes. A binary classification
        problem will have c=1.
    params : python dictionary
        Lists all permutations of hyperparameters, a subset of which will be
        selected at random for training and evaluation.
    dataset_name : str
        References the name of the experiment. The dataset_name and
        experiment_no will be concatenated to produce the file name for the
        results saved in the local directory.
    experiment_no : str
        Indexes the user's choice of experiment number.
    model : keras_model
        A Keras style model which compiles and fits the data, and returns
        the history and compiled model.
    val_split : float, optional
        The proportion of the input `x` which is set aside as the
        cross-validation data. (Default is 0.3).
    shuffle : bool, optional
        If True, shuffle the data in x and y before splitting into the train
        and cross-validation datasets. (Default is True).
    search_method : {None, 'random', 'linear', 'reverse'}
        Determines the random sampling of the dictionary. `random` picks one
        hyperparameter point at random and removes it from the list, then
        samples again. `linear` starts from the start of the grid and moves
        forward, and `reverse` starts at the end of the grid and moves
        backwards.
    reduction_method : {None, 'spear'}
        Method for honing in on the optimal hyperparameter subspace. (Default
        is None).
    reduction_interval : int
        The number of reduction method rounds that will be performed. (Default
        is None).
    reduction_window : int
        The number of rounds of the reduction method before observing the
        results. (Default is None).
    grid_downsample : int
        The fraction of `params` that will be tested (Default is None).
    reduction_metric : {'val_acc'}
        Metric used to tune the reductions.
    talos_log_name : str
        The lame of the saved Talos log. (Default is 'talos.log').
    debug : bool
        Implements debugging feedback. (Default is False).
    x_val : ndarray
        User specified cross-validation data. (Default is None).
    y_val : ndarray
        User specified cross-validation labels. (Default is None).

    """

    # TODO: refactor this so that we don't initialize global variables
    global self

    def __init__(self, x, y, params, dataset_name, experiment_no, model,
                 val_split=.3, shuffle=True, search_method='random',
                 reduction_method=None, reduction_interval=100,
                 reduction_window=None, grid_downsample=None,
                 reduction_metric='val_acc', round_limit=None,
                 talos_log_name='talos.log', debug=False, seed=None,
                 x_val=None, y_val=None, clear_tf_session=False):

        self.dataset_name = dataset_name
        self.experiment_no = experiment_no
        self.experiment_name = dataset_name + '_' + experiment_no

        self.clear_tf_session = clear_tf_session

        self.custom_val_split = False
        if (x_val is not None and y_val is None) or \
           (x_val is None and y_val is not None):
            raise RuntimeError(TRAIN_VAL_RUNTIME_ERROR_MSG)
        elif (x_val is not None and y_val is not None):
            self.custom_val_split = True
            self.x_val = x_val
            self.y_val = y_val

        if debug:
            self.logfile = open('talos.debug.log', 'a')
        else:
            self.logfile_name = talos_log_name
            self.logfile = open(self.logfile_name, 'a')

        self.model = model
        self.param_dict = params

        self.search_method = search_method
        self.grid_downsample = grid_downsample

        self.val_split = val_split
        self.shuffle = shuffle
        self.seed = seed

        # create the paramater object
        self.param_object = ParamGrid(self.param_dict,
                                      self.search_method,
                                      self.grid_downsample)

        # reduction related (main object)
        self.reduction_method = reduction_method

        # reduction related (parameters object)
        self.param_object.reduction_interval = reduction_interval
        self.param_object.reduction_window = reduction_window
        self.param_object.reduction_metric = reduction_metric
        self.param_object.experiment_name = self.experiment_name

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

        # the main runtime starts
        self.result = []
        if self.round_limit is not None:
            for i in range(self.round_limit):
                self._null = self._run()
        else:
            while len(self.param_object.param_log) != 0:
                self._null = self._run()

        self = result_todf(self)
        self.peak_epochs_df = peak_epochs_todf(self)
        self._null = self.logfile.close()
        # runtime ends

        print('Scan Finished!')

    def _run(self):

        # determine the parameters for the particular execution
        self.params = round_params(self.param_object)

        # compile the model
        _hr_out, self.keras_model = self._model()

        # create log and other stats
        self.epoch_entropy.append(epoch_entropy((_hr_out)))

        if self.round_counter == 0:
            _for_header = create_header(self, _hr_out)
            self.result.append(_for_header)
            save_result(self)

        _hr_out = run_round_results(self, _hr_out)
        self._val_score = get_score(self)
        write_log(self)
        self.result.append(_hr_out)
        save_result(self)

        if (self.round_counter + 1) % self.param_object.reduction_interval == 0:
            if self.reduction_method == 'spear':
                self.param_object = spear_reducer(self.param_object)

        if self.clear_tf_session is True:
            K.clear_session()
        self.round_counter += 1

    def _model(self):

        return self.model(self.x_train,
                          self.y_train,
                          self.x_val,
                          self.y_val,
                          self.params)
